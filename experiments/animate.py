from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.labels import (
    COLOR_A,
    COLOR_B,
    COLOR_REVERSE,
    MARK_AT_GOAL,
    TITLE_A,
    TITLE_B,
    VIDEO_METADATA_TITLE,
    decomposition_line,
)
from experiments.viz import _integrate_for_plot, draw_figure_legend, use_publication_style
from models.symmetry import DirectedGoal, GoalManifold, QuotientGoal
from planner.hybrid_astar import HybridAStar
from planner.obstacles import (
    ObstacleMap,
    RectObstacle,
    VEHICLE_LENGTH,
    VEHICLE_WIDTH,
)
from planner.reeds_shepp import Gear, Pose, get_optimal_path, path_length, sample_path

TURNING_RADIUS = 1.0
FPS = 24
DRIVE_SEC = 3.5
HOLD_SEC = 0.6
SPEED = 1.35
N_SCENARIOS = 10


@dataclass
class Scenario:
    index: int
    label: str
    start: Pose
    goal_x: float
    goal_y: float
    goal_theta: float
    obstacles: ObstacleMap
    path_a: tuple
    path_b: tuple
    cost_a: float
    cost_rho1: float
    cost_b: float
    arrival_theta_a: float
    arrival_theta_b: float
    xs_a: np.ndarray
    ys_a: np.ndarray
    th_a: np.ndarray
    xs_b: np.ndarray
    ys_b: np.ndarray
    th_b: np.ndarray
    arc_len_a: float = 0.0
    arc_len_b: float = 0.0
    drive_frames: int = 0
    uses_hybrid: bool = False


def _vehicle_body(x: float, y: float, theta: float) -> np.ndarray:
    half_l, half_w = VEHICLE_LENGTH / 2, VEHICLE_WIDTH / 2
    corners = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ]
    )
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return corners @ rot.T + np.array([x, y])


def _front_dot(x: float, y: float, theta: float) -> tuple[float, float]:
    return x + 0.42 * math.cos(theta), y + 0.42 * math.sin(theta)


def _arc_length(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))


def _pose_at_arc(
    xs: np.ndarray, ys: np.ndarray, thetas: np.ndarray, s: float
) -> tuple[float, float, float]:
    if len(xs) == 0:
        return 0.0, 0.0, 0.0
    if len(xs) == 1 or s <= 0.0:
        return float(xs[0]), float(ys[0]), float(thetas[0])
    ds = np.hypot(np.diff(xs), np.diff(ys))
    cum = np.concatenate([[0.0], np.cumsum(ds)])
    if s >= cum[-1]:
        return float(xs[-1]), float(ys[-1]), float(thetas[-1])
    th_u = np.unwrap(thetas)
    return (
        float(np.interp(s, cum, xs)),
        float(np.interp(s, cum, ys)),
        float(np.interp(s, cum, th_u)),
    )


def _trail_until(
    xs: np.ndarray, ys: np.ndarray, s: float
) -> tuple[np.ndarray, np.ndarray]:
    if len(xs) < 2:
        return xs, ys
    ds = np.hypot(np.diff(xs), np.diff(ys))
    cum = np.concatenate([[0.0], np.cumsum(ds)])
    idx = int(np.searchsorted(cum, s, side="right"))
    idx = max(1, min(idx, len(xs)))
    tx, ty = xs[:idx].copy(), ys[:idx].copy()
    x, y, _ = _pose_at_arc(xs, ys, np.zeros(len(xs)), s)
    if len(tx) == 0 or math.hypot(x - tx[-1], y - ty[-1]) > 1e-4:
        tx = np.append(tx, x)
        ty = np.append(ty, y)
    return tx, ty


def _trajectory_segments(
    xs: np.ndarray, ys: np.ndarray, thetas: np.ndarray, color: str
) -> list[tuple[np.ndarray, str, str]]:
    segments: list[tuple[np.ndarray, str, str]] = []
    for i in range(len(xs) - 1):
        dx = float(xs[i + 1] - xs[i])
        dy = float(ys[i + 1] - ys[i])
        if dx * dx + dy * dy < 1e-10:
            continue
        motion = math.atan2(dy, dx)
        dtheta = abs((motion - float(thetas[i]) + math.pi) % (2 * math.pi) - math.pi)
        forward = dtheta <= math.pi / 2
        ls = "-" if forward else "--"
        col = color if forward else COLOR_REVERSE
        segments.append(
            (np.array([[xs[i], ys[i]], [xs[i + 1], ys[i + 1]]]), ls, col)
        )
    return segments


def _path_segments(
    start: Pose, path, color: str
) -> list[tuple[np.ndarray, str, str]]:
    segments: list[tuple[np.ndarray, str, str]] = []
    x, y, th = start.x, start.y, start.theta
    for elem in path:
        seg_xs, seg_ys, _, x, y, th = _integrate_for_plot(x, y, th, elem, 0.03)
        pts = np.column_stack([seg_xs, seg_ys])
        ls = "--" if elem.gear is Gear.BACKWARD else "-"
        col = COLOR_REVERSE if elem.gear is Gear.BACKWARD else color
        segments.append((pts, ls, col))
    return segments


def _trajectory_stats(
    xs: np.ndarray, ys: np.ndarray, thetas: np.ndarray
) -> tuple[int, float, bool]:
    if len(xs) < 2:
        return 0, 0.0, True
    rev_dist = 0.0
    total = 0.0
    for i in range(len(xs) - 1):
        dx = float(xs[i + 1] - xs[i])
        dy = float(ys[i + 1] - ys[i])
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            continue
        motion = math.atan2(dy, dx)
        dtheta = abs((motion - float(thetas[i]) + math.pi) % (2 * math.pi) - math.pi)
        if dtheta > math.pi / 2:
            rev_dist += dist
        total += dist
    n = max(1, int(len(xs) / 8))
    rev_frac = rev_dist / total if total > 0 else 0.0
    straight = len(xs) < 25 and rev_frac < 0.05
    return n, rev_frac, straight


def _path_stats(path) -> tuple[int, float, bool]:
    n = len(path)
    rev = sum(abs(e.length) for e in path if e.gear is Gear.BACKWARD)
    total = sum(abs(e.length) for e in path) or 1.0
    straight = n == 1 and path[0].steering.value == "S"
    return n, rev / total, straight


def _stats(sc: Scenario, side: str) -> tuple[int, float, bool]:
    if sc.uses_hybrid:
        xs = sc.xs_a if side == "a" else sc.xs_b
        ys = sc.ys_a if side == "a" else sc.ys_b
        th = sc.th_a if side == "a" else sc.th_b
        return _trajectory_stats(xs, ys, th)
    path = sc.path_a if side == "a" else sc.path_b
    return _path_stats(path)


def _trajectory_clear(
    xs: np.ndarray, ys: np.ndarray, thetas: np.ndarray, obstacles: ObstacleMap
) -> bool:
    if len(xs) < 2:
        return not obstacles.vehicle_collision(float(xs[0]), float(ys[0]), float(thetas[0]))
    for i in range(len(xs) - 1):
        seg_len = float(np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]))
        n = max(2, int(math.ceil(seg_len / 0.08)))
        th_u = np.unwrap(thetas)
        for t in np.linspace(0.0, 1.0, n, endpoint=True):
            x = float(xs[i] + t * (xs[i + 1] - xs[i]))
            y = float(ys[i] + t * (ys[i + 1] - ys[i]))
            th = float(th_u[i] + t * (th_u[i + 1] - th_u[i]))
            if obstacles.vehicle_collision(x, y, th):
                return False
    return True


def _hybrid_plan(
    start: Pose,
    goal: GoalManifold,
    *,
    reverse_penalty: float,
    obstacles: ObstacleMap,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    planner = HybridAStar(
        goal=goal,
        turning_radius=TURNING_RADIUS,
        reverse_penalty=reverse_penalty,
        collision_checker=obstacles.vehicle_collision,
        xy_resolution=0.3,
        step_size=0.3,
        analytic_radius=7.0,
    )
    result = planner.plan(start)
    if result is None:
        raise RuntimeError("Hybrid A* failed to find a path")
    if not _trajectory_clear(result.xs, result.ys, result.thetas, obstacles):
        raise RuntimeError("Planned path intersects obstacles")
    return (
        result.xs,
        result.ys,
        result.thetas,
        result.cost,
        float(result.thetas[-1]),
    )


def _build_scenario(
    index: int,
    start: Pose,
    gx: float,
    gy: float,
    gtheta: float,
    *,
    axis_theta: float | None = None,
    label: str = "",
    obstacles: ObstacleMap | None = None,
    prepare: bool = True,
) -> Scenario:
    directed_theta = gtheta
    axis = gtheta if axis_theta is None else axis_theta
    obstacles = obstacles or ObstacleMap()
    goal_a = DirectedGoal(
        gx, gy, directed_theta,
        turning_radius=TURNING_RADIUS,
        reverse_penalty=2.0,
    )
    goal_rho1 = DirectedGoal(
        gx, gy, directed_theta,
        turning_radius=TURNING_RADIUS,
        reverse_penalty=1.0,
    )
    goal_b = QuotientGoal(
        gx, gy, axis,
        turning_radius=TURNING_RADIUS,
        reverse_penalty=1.0,
    )

    if obstacles.rects:
        if obstacles.vehicle_collision(start.x, start.y, start.theta):
            raise RuntimeError("Start pose collides with obstacles")
        if obstacles.vehicle_collision(gx, gy, directed_theta):
            raise RuntimeError("Goal pose collides with obstacles")
        xs_a, ys_a, th_a, cost_a, arrival_theta_a = _hybrid_plan(
            start, goal_a, reverse_penalty=2.0, obstacles=obstacles
        )
        _, _, _, cost_rho1, _ = _hybrid_plan(
            start, goal_rho1, reverse_penalty=1.0, obstacles=obstacles
        )
        xs_b, ys_b, th_b, cost_b, arrival_theta_b = _hybrid_plan(
            start, goal_b, reverse_penalty=1.0, obstacles=obstacles
        )
        path_a, path_b = (), ()
        uses_hybrid = True
    else:
        path_a = get_optimal_path(
            start, goal_a._goal_pose(), TURNING_RADIUS, reverse_penalty=2.0
        )
        analytic_b = goal_b.analytic_expansion(start.x, start.y, start.theta)
        assert analytic_b is not None
        path_b = analytic_b.path
        cost_a = path_length(path_a, 2.0)
        cost_rho1 = goal_rho1.heuristic(start.x, start.y, start.theta)
        cost_b = analytic_b.cost
        arrival_theta_a = directed_theta
        arrival_theta_b = analytic_b.goal_theta
        xs_a = ys_a = th_a = xs_b = ys_b = th_b = np.array([])
        uses_hybrid = False

    sc = Scenario(
        index=index,
        label=label,
        start=start,
        goal_x=gx,
        goal_y=gy,
        goal_theta=directed_theta,
        obstacles=obstacles,
        path_a=path_a,
        path_b=path_b,
        cost_a=cost_a,
        cost_rho1=cost_rho1,
        cost_b=cost_b,
        arrival_theta_a=arrival_theta_a,
        arrival_theta_b=arrival_theta_b,
        xs_a=xs_a,
        ys_a=ys_a,
        th_a=th_a,
        xs_b=xs_b,
        ys_b=ys_b,
        th_b=th_b,
        uses_hybrid=uses_hybrid,
    )
    if prepare:
        return _prepare_paths(sc)
    return sc


def _prepare_paths(sc: Scenario) -> Scenario:
    if not sc.uses_hybrid:
        xs_a, ys_a, th_a, _ = sample_path(sc.start, sc.path_a, step=0.02)
        xs_b, ys_b, th_b, _ = sample_path(sc.start, sc.path_b, step=0.02)
        sc.xs_a, sc.ys_a, sc.th_a = xs_a, ys_a, th_a
        sc.xs_b, sc.ys_b, sc.th_b = xs_b, ys_b, th_b
    sc.arc_len_a = _arc_length(sc.xs_a, sc.ys_a)
    sc.arc_len_b = _arc_length(sc.xs_b, sc.ys_b)
    max_len = max(sc.arc_len_a, sc.arc_len_b)
    min_drive = int(FPS * DRIVE_SEC * 0.35)
    sc.drive_frames = max(
        min_drive,
        int(math.ceil(max_len / (SPEED / FPS))),
    )
    return sc


def _scenario_bounds(sc: Scenario, margin: float = 1.0) -> tuple[float, float, float, float]:
    xs = np.concatenate([sc.xs_a, sc.xs_b, [sc.start.x, sc.goal_x]])
    ys = np.concatenate([sc.ys_a, sc.ys_b, [sc.start.y, sc.goal_y]])
    if sc.obstacles.rects:
        ob = sc.obstacles.bounds()
        if ob is not None:
            xs = np.concatenate([xs, [ob[0], ob[1]]])
            ys = np.concatenate([ys, [ob[2], ob[3]]])
    return float(xs.min()) - margin, float(xs.max()) + margin, float(ys.min()) - margin, float(ys.max()) + margin


def _classify(sc: Scenario) -> str:
    delta_rho = sc.cost_a - sc.cost_rho1
    delta_theta = sc.cost_rho1 - sc.cost_b
    dist = math.hypot(sc.start.x - sc.goal_x, sc.start.y - sc.goal_y)
    na, rev_a, straight_a = _stats(sc, "a")
    nb, rev_b, straight_b = _stats(sc, "b")
    used_opposite = abs(
        (sc.arrival_theta_b - sc.goal_theta + math.pi) % (2 * math.pi) - math.pi
    ) < 0.05

    if dist < 3.0:
        if delta_rho < 0.08 and delta_theta < 0.08:
            return "tight_similar"
        return "tight_maneuver"

    if delta_rho < 0.06 and delta_theta < 0.06:
        return "near_equal"

    if delta_rho >= 0.12 and delta_theta < 0.06:
        return "rho_only"

    if delta_theta >= 0.12 and delta_rho < 0.06:
        return "axis_only"

    if straight_b and not straight_a and delta_theta >= 0.15:
        return "straight_vs_loop"

    if na >= 3 and nb >= 3 and not straight_b and not straight_a:
        return "both_complex"

    if used_opposite and delta_theta >= 0.08:
        return "opposite_axis"

    if delta_rho >= 0.10 and delta_theta >= 0.10:
        return "both_effects"

    if rev_b < 0.15 and rev_a > 0.25:
        return "forward_vs_reverse"

    return "general"


_CURATED_DESCRIPTIONS: dict[str, str] = {
    "reverse_vs_loop": "Goal behind a wall: A loops forward, B reverses through the open lane",
    "axis_parking": "Parking slot between curbs, axis goal picks the shorter entry heading",
    "rho_only": "Same detour around a central block, reverse penalty is the only difference",
    "near_goal_wrong_nose": "Wrong nose at the goal, axis goal avoids a long final loop",
    "long_diagonal": "Diagonal transfer detouring around two separated blocks",
    "obstacle_detour": "Central pillar: A pays for directed θ*, B detours with axis heading",
    "both_effects": "Vertical transfer where ρ and axis goal both reduce cost",
    "axis_only": "Side block leaves a forward lane; axis goal avoids the final heading loop",
    "forward_vs_reverse": "Lower lane stays open for backing, upper lane blocked",
    "near_equal": "Small offset block, both planners take similar detours",
}


def _scene_description(sc: Scenario) -> str:
    key = sc.label or _classify(sc)
    base = _CURATED_DESCRIPTIONS.get(key, "Curated obstacle scene")
    if sc.obstacles.rects:
        base += " (with obstacles)"
    return base


def _wall(x0: float, y0: float, x1: float, y1: float) -> RectObstacle:
    return RectObstacle(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


CURATED: list[tuple[str, Pose, float, float, float, float | None, ObstacleMap]] = [
    (
        "reverse_vs_loop",
        Pose(0.0, 0.0, 0.0),
        -4.0, 0.0, math.pi, 0.0,
        ObstacleMap((
            _wall(-3.5, 0.45, -0.6, 2.4),
        )),
    ),
    (
        "axis_parking",
        Pose(0.0, 0.0, 0.0),
        4.2, 0.0, math.pi, 0.0,
        ObstacleMap((
            _wall(1.6, -1.5, 5.0, -0.55),
            _wall(1.6, 0.55, 5.0, 1.5),
        )),
    ),
    (
        "rho_only",
        Pose(0.0, 0.0, 0.0),
        -5.5, 0.0, 0.0, 0.0,
        ObstacleMap((
            _wall(-3.0, -0.25, -2.0, 0.25),
        )),
    ),
    (
        "near_goal_wrong_nose",
        Pose(0.5, 0.0, 0.0),
        3.8, 0.0, math.pi, 0.0,
        ObstacleMap((
            _wall(2.2, -0.6, 2.7, 0.6),
        )),
    ),
    (
        "long_diagonal",
        Pose(-5.0, -4.0, 0.6),
        5.0, 4.0, -2.0, None,
        ObstacleMap((
            _wall(-1.5, -2.5, 0.5, -1.0),
            _wall(1.0, 1.2, 2.5, 2.2),
        )),
    ),
    (
        "obstacle_detour",
        Pose(-3.5, 0.0, 0.0),
        3.5, 0.0, math.pi, 0.0,
        ObstacleMap((
            _wall(-0.45, -0.45, 0.45, 0.45),
        )),
    ),
    (
        "both_effects",
        Pose(0.0, -2.8, math.pi / 2),
        0.0, 2.8, -math.pi / 2, 0.0,
        ObstacleMap((
            _wall(-0.55, -0.55, 0.55, 0.55),
            _wall(-2.4, 0.9, -0.75, 2.2),
        )),
    ),
    (
        "axis_only",
        Pose(0.0, 0.0, 0.0),
        4.0, 0.0, math.pi / 2, 0.0,
        ObstacleMap((
            _wall(1.5, -1.2, 2.8, -0.2),
        )),
    ),
    (
        "forward_vs_reverse",
        Pose(0.0, 0.0, 0.0),
        -3.5, 0.0, 0.0, 0.0,
        ObstacleMap((
            _wall(-2.0, 0.45, -0.6, 2.0),
        )),
    ),
    (
        "near_equal",
        Pose(-2.0, -2.0, 0.4),
        2.0, 2.0, -1.2, None,
        ObstacleMap((
            _wall(-0.35, -0.35, 0.35, 0.35),
        )),
    ),
]


def select_scenarios(n: int = N_SCENARIOS) -> list[Scenario]:
    out: list[Scenario] = []
    for i, (key, start, gx, gy, gt, axis, obstacles) in enumerate(CURATED[:n], start=1):
        sc = _build_scenario(
            i, start, gx, gy, gt,
            axis_theta=axis, label=key, obstacles=obstacles,
        )
        out.append(sc)

    for sc in out:
        delta_rho = sc.cost_a - sc.cost_rho1
        delta_theta = sc.cost_rho1 - sc.cost_b
        print(
            f"  {sc.index:2d}.  "
            f"$J^*(A)={sc.cost_a:.2f}$  $J^*(B)={sc.cost_b:.2f}$  "
            f"$\\Delta_\\rho={delta_rho:.2f}$  $\\Delta_\\theta={delta_theta:.2f}$"
        )
    return out


def render_video(out_path: Path, scenarios: list[Scenario] | None = None) -> None:
    use_publication_style()
    scenarios = scenarios or select_scenarios()
    hold_frames = int(FPS * HOLD_SEC)
    ds_per_frame = SPEED / FPS

    frames_per_list = [
        hold_frames + sc.drive_frames + hold_frames for sc in scenarios
    ]
    total_frames = sum(frames_per_list)
    frame_offsets = np.cumsum([0] + frames_per_list[:-1])

    fig = plt.figure(figsize=(12, 8.0))
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[0.85, 4.2, 1.45],
        hspace=0.62,
        wspace=0.22,
        left=0.07,
        right=0.98,
        top=0.94,
        bottom=0.08,
    )
    header_ax = fig.add_subplot(gs[0, :])
    header_ax.axis("off")
    ax_a = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[1, 1])
    draw_figure_legend(fig.add_subplot(gs[2, :]))

    header_desc = header_ax.text(
        0.5, 0.97, "", ha="center", va="top", fontsize=9.5, color="#555555", style="italic",
        transform=header_ax.transAxes,
    )
    header_pose = header_ax.text(
        0.5, 0.52, "", ha="center", va="center", fontsize=11, fontweight="bold",
        transform=header_ax.transAxes,
    )
    header_delta = header_ax.text(
        0.5, 0.08, "", ha="center", va="center", fontsize=9, color="#444444",
        transform=header_ax.transAxes,
    )

    poly_a = Polygon(_vehicle_body(0, 0, 0), closed=True, fc=COLOR_A, ec="white", lw=1.2, zorder=10)
    poly_b = Polygon(_vehicle_body(0, 0, 0), closed=True, fc=COLOR_B, ec="white", lw=1.2, zorder=10)
    ax_a.add_patch(poly_a)
    ax_b.add_patch(poly_b)
    (dot_a,) = ax_a.plot([], [], "o", color="white", ms=5, zorder=11)
    (dot_b,) = ax_b.plot([], [], "o", color="white", ms=4, alpha=0.55, zorder=11)

    trail_a, = ax_a.plot([], [], color=COLOR_A, lw=1.5, alpha=0.35, zorder=5)
    trail_b, = ax_b.plot([], [], color=COLOR_B, lw=1.5, alpha=0.35, zorder=5)
    status_b = ax_b.text(
        0.97, 0.97, "", transform=ax_b.transAxes, ha="right", va="top", fontsize=9,
        color=COLOR_B, fontweight="bold",
    )
    cost_text_a = ax_a.text(
        0.03, 0.03, "", transform=ax_a.transAxes, fontsize=9, va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.92, edgecolor="#cccccc"),
    )
    cost_text_b = ax_b.text(
        0.03, 0.03, "", transform=ax_b.transAxes, fontsize=9, va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.92, edgecolor="#cccccc"),
    )

    static_artists: list = []

    def setup_scenario(sc: Scenario) -> None:
        for artist in static_artists:
            artist.remove()
        static_artists.clear()
        trail_a.set_data([], [])
        trail_b.set_data([], [])

        xmin, xmax, ymin, ymax = _scenario_bounds(sc)
        for ax in (ax_a, ax_b):
            ax.set_aspect("equal")
            ax.grid(True)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        ax_a.set_title(TITLE_A, fontsize=10, fontweight="bold", pad=12)
        ax_b.set_title(TITLE_B, fontsize=10, fontweight="bold", pad=12)

        for ax in (ax_a, ax_b):
            static_artists.extend(sc.obstacles.draw(ax))

        if sc.uses_hybrid:
            path_draw = (
                (ax_a, _trajectory_segments(sc.xs_a, sc.ys_a, sc.th_a, COLOR_A)),
                (ax_b, _trajectory_segments(sc.xs_b, sc.ys_b, sc.th_b, COLOR_B)),
            )
        else:
            path_draw = (
                (ax_a, _path_segments(sc.start, sc.path_a, COLOR_A)),
                (ax_b, _path_segments(sc.start, sc.path_b, COLOR_B)),
            )
        for ax, segs in path_draw:
            for pts, ls, col in segs:
                (ln,) = ax.plot(pts[:, 0], pts[:, 1], ls=ls, color=col, lw=1.8, alpha=0.55, zorder=3)
                static_artists.append(ln)

        for ax, color, th, show_nose in (
            (ax_a, COLOR_A, sc.arrival_theta_a, True),
            (ax_b, COLOR_B, sc.arrival_theta_b, False),
        ):
            (gm,) = ax.plot(sc.goal_x, sc.goal_y, "o", color=color, ms=6, zorder=6)
            static_artists.append(gm)
            gx, gy = sc.goal_x, sc.goal_y
            body = _vehicle_body(gx, gy, th)
            ghost = Polygon(
                body, closed=True, fill=False, edgecolor=color, lw=1.2, alpha=0.3, zorder=4
            )
            ax.add_patch(ghost)
            static_artists.append(ghost)
            if show_nose:
                fx, fy = _front_dot(gx, gy, th)
                (nose,) = ax.plot(fx, fy, "o", color=color, ms=4, alpha=0.35, zorder=5)
                static_artists.append(nose)

        status_b.set_text("")

        cost_text_a.set_text(
            rf"$J^*={sc.cost_a:.2f}$" + "\n"
            rf"directed $\theta^*$"
        )
        cost_text_b.set_text(
            rf"$J^*={sc.cost_b:.2f}$" + "\n"
            rf"$\theta={sc.arrival_theta_b:.2f}$"
        )

        delta_rho = sc.cost_a - sc.cost_rho1
        delta_theta = sc.cost_rho1 - sc.cost_b
        header_desc.set_text(_scene_description(sc))
        header_pose.set_text(
            rf"$(x,y,\theta)_0=({sc.start.x:.1f},{sc.start.y:.1f},{sc.start.theta:.2f})"
            rf" \to (x,y,\theta^*)=({sc.goal_x:.1f},{sc.goal_y:.1f},{sc.goal_theta:.2f})$"
            f"   [{sc.index}/{len(scenarios)}]"
        )
        header_delta.set_text(decomposition_line(delta_rho, delta_theta))

    def frame_index(global_i: int) -> tuple[int, int, str]:
        si = int(np.searchsorted(frame_offsets, global_i, side="right") - 1)
        si = max(0, min(si, len(scenarios) - 1))
        local = global_i - int(frame_offsets[si])
        drive_n = scenarios[si].drive_frames
        if local < hold_frames:
            return si, 0, "hold_start"
        if local < hold_frames + drive_n:
            return si, local - hold_frames, "drive"
        return si, drive_n - 1, "hold_end"

    def draw(global_i: int) -> None:
        si, fi, phase = frame_index(global_i)
        sc = scenarios[si]
        if fi == 0 and phase == "hold_start":
            setup_scenario(sc)

        if phase == "hold_start":
            s = 0.0
        elif phase == "hold_end":
            s = sc.drive_frames * ds_per_frame
        else:
            s = (fi + 1) * ds_per_frame

        xa, ya, ta = _pose_at_arc(sc.xs_a, sc.ys_a, sc.th_a, s)
        xb, yb, tb = _pose_at_arc(sc.xs_b, sc.ys_b, sc.th_b, s)

        poly_a.set_xy(_vehicle_body(xa, ya, ta))
        poly_b.set_xy(_vehicle_body(xb, yb, tb))
        fxa, fya = _front_dot(xa, ya, ta)
        dot_a.set_data([fxa], [fya])
        dot_b.set_data([], [])

        tx_a, ty_a = _trail_until(sc.xs_a, sc.ys_a, s)
        tx_b, ty_b = _trail_until(sc.xs_b, sc.ys_b, s)
        trail_a.set_data(tx_a, ty_a)
        trail_b.set_data(tx_b, ty_b)

        if s >= sc.arc_len_b - 0.05 and sc.arc_len_b < sc.arc_len_a - 0.1:
            status_b.set_text(MARK_AT_GOAL)
        else:
            status_b.set_text("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=FPS, bitrate=3600, metadata={"title": VIDEO_METADATA_TITLE})

    print(f"Rendering {len(scenarios)} scenarios, {total_frames} frames @ {FPS} fps …")
    with writer.saving(fig, str(out_path), dpi=120):
        for i in range(total_frames):
            draw(i)
            writer.grab_frame()
            if i % FPS == 0:
                print(f"  frame {i}/{total_frames}")

    plt.close(fig)
    print(f"Saved video → {out_path}")


if __name__ == "__main__":
    render_video(ROOT / "output" / "compare.mp4")
