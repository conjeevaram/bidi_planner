from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from models.symmetry import AnalyticResult, DirectedGoal, GoalManifold, QuotientGoal
from planner.reeds_shepp import Gear, Pose, sample_path


@dataclass(order=True)
class _HeapNode:
    f: float
    g: float
    ix: int = field(compare=False)
    iy: int = field(compare=False)
    itheta: int = field(compare=False)


@dataclass
class PlanResult:
    cost: float
    xs: np.ndarray
    ys: np.ndarray
    thetas: np.ndarray
    reverse_segments: list[tuple[int, int]]
    expanded_nodes: int
    analytic: bool


@dataclass
class HybridAStar:
    goal: GoalManifold
    turning_radius: float = 1.0
    xy_resolution: float = 0.5
    theta_resolution: float = math.pi / 12.0
    step_size: float = 0.5
    max_steering: float = math.pi / 4.0
    n_steer_samples: int = 5
    reverse_penalty: float = 1.0
    analytic_radius: float = 6.0
    max_iterations: int = 50_000
    collision_checker: Callable[[float, float, float], bool] | None = None

    def __post_init__(self) -> None:
        self.n_theta = max(1, int(round(2 * math.pi / self.theta_resolution)))
        self.theta_resolution = 2 * math.pi / self.n_theta
        self.motion_primitives = self._build_motion_primitives()

    def _theta_index(self, theta: float) -> int:
        t = (theta + math.pi) % (2 * math.pi) - math.pi
        idx = int(round((t + math.pi) / self.theta_resolution)) % self.n_theta
        return idx

    def _theta_from_index(self, itheta: int) -> float:
        return -math.pi + (itheta + 0.5) * self.theta_resolution

    def _cell(self, x: float, y: float, theta: float) -> tuple[int, int, int]:
        return (
            int(round(x / self.xy_resolution)),
            int(round(y / self.xy_resolution)),
            self._theta_index(theta),
        )

    def _build_motion_primitives(self) -> list[tuple[float, float, float]]:
        primitives: list[tuple[float, float, float]] = []
        steerings = np.linspace(-self.max_steering, self.max_steering, self.n_steer_samples)
        for steer in steerings:
            for direction in (1.0, -1.0):
                dist = self.step_size
                rho = self.reverse_penalty if direction < 0 else 1.0
                primitives.append((float(steer), direction * dist, rho * dist))
        return primitives

    def _is_collision_free(self, x: float, y: float, theta: float) -> bool:
        if self.collision_checker is None:
            return True
        return not self.collision_checker(x, y, theta)

    def _segment_collision_free(
        self, x0: float, y0: float, theta0: float, steer: float, dist: float
    ) -> tuple[bool, float, float, float]:
        x, y, theta = x0, y0, theta0
        n_checks = max(1, int(math.ceil(abs(dist) / (self.step_size * 0.5))))
        ds = dist / n_checks
        for _ in range(n_checks):
            if abs(steer) < 1e-6:
                x += ds * math.cos(theta)
                y += ds * math.sin(theta)
            else:
                radius = self.turning_radius / math.tan(steer)
                dtheta = ds / radius
                x += radius * (math.sin(theta + dtheta) - math.sin(theta))
                y += radius * (-math.cos(theta + dtheta) + math.cos(theta))
                theta += dtheta
            if not self._is_collision_free(x, y, theta):
                return False, x, y, theta
        return True, x, y, theta

    def _append_motion_segment(
        self,
        xs: list[float],
        ys: list[float],
        thetas: list[float],
        x0: float,
        y0: float,
        theta0: float,
        steer: float,
        dist: float,
        *,
        sample_step: float = 0.08,
    ) -> tuple[float, float, float]:
        x, y, theta = x0, y0, theta0
        n_steps = max(1, int(math.ceil(abs(dist) / sample_step)))
        ds = dist / n_steps
        for _ in range(n_steps):
            if abs(steer) < 1e-6:
                x += ds * math.cos(theta)
                y += ds * math.sin(theta)
            else:
                radius = self.turning_radius / math.tan(steer)
                dtheta = ds / radius
                x += radius * (math.sin(theta + dtheta) - math.sin(theta))
                y += radius * (-math.cos(theta + dtheta) + math.cos(theta))
                theta += dtheta
            xs.append(x)
            ys.append(y)
            thetas.append(theta)
        return x, y, theta

    def _path_from_parent_chain(
        self,
        parent: dict[tuple[int, int, int], tuple[int, int, int] | None],
        state: tuple[int, int, int],
        state_pose: dict[tuple[int, int, int], tuple[float, float, float]],
        edge_motion: dict[tuple[int, int, int], tuple[float, float]],
        analytic_tail: AnalyticResult | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
        chain: list[tuple[int, int, int]] = []
        cur = state
        while cur is not None:
            chain.append(cur)
            cur = parent.get(cur)
        chain.reverse()

        xs: list[float] = []
        ys: list[float] = []
        thetas: list[float] = []
        reverse_segments: list[tuple[int, int]] = []

        x0, y0, th0 = state_pose[chain[0]]
        xs.append(x0)
        ys.append(y0)
        thetas.append(th0)

        for cell in chain[1:]:
            steer, dist = edge_motion[cell]
            if dist < 0:
                seg_start = len(xs)
            self._append_motion_segment(xs, ys, thetas, x0, y0, th0, steer, dist)
            if dist < 0:
                reverse_segments.append((seg_start, len(xs) - 1))
            x0, y0, th0 = xs[-1], ys[-1], thetas[-1]

        if analytic_tail is not None:
            start = Pose(xs[-1], ys[-1], thetas[-1])
            ax, ay, ath, _ = sample_path(start, analytic_tail.path, step=0.08)
            base = len(xs)
            for i, elem in enumerate(analytic_tail.path):
                if elem.gear is Gear.BACKWARD:
                    seg_start = base + int(i * len(ax) / max(len(analytic_tail.path), 1))
                    reverse_segments.append((seg_start, len(ax) + base - 1))
            xs.extend(ax[1:].tolist())
            ys.extend(ay[1:].tolist())
            thetas.extend(ath[1:].tolist())

        return np.asarray(xs), np.asarray(ys), np.asarray(thetas), reverse_segments

    def plan(self, start: Pose) -> PlanResult | None:
        start_cell = self._cell(start.x, start.y, start.theta)
        g_score: dict[tuple[int, int, int], float] = {start_cell: 0.0}
        parent: dict[tuple[int, int, int], tuple[int, int, int] | None] = {
            start_cell: None
        }
        edge_motion: dict[tuple[int, int, int], tuple[float, float]] = {}
        state_pose: dict[tuple[int, int, int], tuple[float, float, float]] = {
            start_cell: (start.x, start.y, start.theta)
        }
        closed: set[tuple[int, int, int]] = set()
        open_heap: list[_HeapNode] = []
        h0 = self.goal.heuristic(start.x, start.y, start.theta)
        heapq.heappush(open_heap, _HeapNode(h0, 0.0, *start_cell))

        expanded = 0
        while open_heap and expanded < self.max_iterations:
            node = heapq.heappop(open_heap)
            cell = (node.ix, node.iy, node.itheta)
            if cell in closed:
                continue
            closed.add(cell)
            expanded += 1

            x, y, theta = state_pose[cell]
            g = g_score[cell]

            if self.goal.is_goal(x, y, theta):
                xs, ys, thetas, rev = self._path_from_parent_chain(
                    parent, cell, state_pose, edge_motion
                )
                return PlanResult(g, xs, ys, thetas, rev, expanded, False)

            if self.goal.heuristic(x, y, theta) <= self.analytic_radius:
                analytic = self.goal.analytic_expansion(x, y, theta)
                if analytic is not None and self._analytic_collision_free(
                    x, y, theta, analytic
                ):
                    total = g + analytic.cost
                    xs, ys, thetas, rev = self._path_from_parent_chain(
                        parent, cell, state_pose, edge_motion, analytic
                    )
                    return PlanResult(total, xs, ys, thetas, rev, expanded, True)

            for steer, dist, step_cost in self.motion_primitives:
                ok, nx, ny, ntheta = self._segment_collision_free(
                    x, y, theta, steer, dist
                )
                if not ok:
                    continue
                ncell = self._cell(nx, ny, ntheta)
                if ncell in closed:
                    continue
                tentative = g + step_cost
                if tentative < g_score.get(ncell, float("inf")):
                    g_score[ncell] = tentative
                    parent[ncell] = cell
                    edge_motion[ncell] = (steer, dist)
                    state_pose[ncell] = (nx, ny, ntheta)
                    h = self.goal.heuristic(nx, ny, ntheta)
                    heapq.heappush(
                        open_heap,
                        _HeapNode(tentative + h, tentative, *ncell),
                    )

        return None

    def _analytic_collision_free(
        self, x: float, y: float, theta: float, analytic: AnalyticResult
    ) -> bool:
        start = Pose(x, y, theta)
        xs, ys, thetas, _ = sample_path(start, analytic.path, step=0.08)
        for px, py, pth in zip(xs, ys, thetas):
            if not self._is_collision_free(float(px), float(py), float(pth)):
                return False
        return True


def make_goal(
    x: float,
    y: float,
    theta: float,
    *,
    quotient: bool = False,
    turning_radius: float = 1.0,
    reverse_penalty: float = 1.0,
) -> GoalManifold:
    cls = QuotientGoal if quotient else DirectedGoal
    return cls(
        x,
        y,
        theta,
        turning_radius=turning_radius,
        reverse_penalty=reverse_penalty,
    )


def plan_free_space(
    start: Pose,
    goal: GoalManifold,
) -> PlanResult:
    analytic = goal.analytic_expansion(start.x, start.y, start.theta)
    if analytic is None:
        raise RuntimeError("No analytic path to goal")
    xs, ys, thetas, _ = sample_path(
        start, analytic.path, step=0.05
    )
    reverse_segments: list[tuple[int, int]] = []
    idx = 0
    for elem in analytic.path:
        n_pts = max(2, int(abs(elem.length) / 0.05))
        if elem.gear is Gear.BACKWARD:
            reverse_segments.append((idx, idx + n_pts))
        idx += n_pts
    return PlanResult(
        analytic.cost, xs, ys, thetas, reverse_segments, 0, True
    )
