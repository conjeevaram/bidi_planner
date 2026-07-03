from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from experiments.labels import (
    CBAR_TOTAL,
    COLOR_A,
    COLOR_B,
    COLOR_REVERSE,
    LEGEND_A,
    LEGEND_B,
    LEGEND_J,
    OVERVIEW_TITLE,
    TITLE_A,
    TITLE_B,
    XLABEL_RHO,
    YLABEL_GOAL,
)
from models.symmetry import DirectedGoal, QuotientGoal
from planner.reeds_shepp import Gear, Pose, path_length


def use_publication_style() -> None:
    plt.rcParams.update(
        {
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.35,
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "standard",
        }
    )


def _integrate_for_plot(x, y, theta, elem, step):
    from planner.reeds_shepp import _integrate_segment

    return _integrate_segment(x, y, theta, elem, step)


def draw_figure_legend(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("none")

    y_sym, y_a, y_b, y_j = 0.92, 0.64, 0.42, 0.16

    ax.plot([0.06, 0.13], [y_sym, y_sym], color=COLOR_A, lw=3.5, solid_capstyle="round")
    ax.text(0.145, y_sym, "forward", va="center", fontsize=9)

    ax.plot(
        [0.24, 0.31], [y_sym, y_sym],
        color=COLOR_REVERSE, lw=3.5, ls=(0, (5, 3)), solid_capstyle="round",
    )
    ax.text(0.325, y_sym, "reverse", va="center", fontsize=9, color=COLOR_REVERSE)

    ax.plot(
        0.43, y_sym, "o", color="white", markeredgecolor="#333",
        markeredgewidth=1.5, markersize=8, zorder=3,
    )
    ax.text(0.445, y_sym, "heading", va="center", fontsize=9)

    ax.plot(
        0.55, y_sym, "o", markerfacecolor="none", markeredgecolor="#333",
        markeredgewidth=1.5, markersize=8,
    )
    ax.text(0.565, y_sym, "goal", va="center", fontsize=9)

    ax.plot(
        0.03, y_a, "s", color=COLOR_A, markersize=9,
        markeredgecolor="white", markeredgewidth=0.6, clip_on=False,
    )
    ax.text(0.045, y_a, LEGEND_A, va="center", ha="left", fontsize=8.5)

    ax.plot(
        0.03, y_b, "s", color=COLOR_B, markersize=9,
        markeredgecolor="white", markeredgewidth=0.6, clip_on=False,
    )
    ax.text(0.045, y_b, LEGEND_B, va="center", ha="left", fontsize=8.5)

    ax.text(0.5, y_j, LEGEND_J, ha="center", va="center", fontsize=8.5, color="#444444")


def draw_stack_pipeline(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Hybrid A* + Reeds-Shepp", fontsize=10, pad=10)

    boxes = [
        (0.4, 6.8, 2.6, 1.6, "Start\n$(x,y,\\theta)$", "#eef2ff"),
        (3.6, 6.8, 2.8, 1.6, "Hybrid A*\n$(x,y,\\theta)$ lattice", "#e8f4fc"),
        (7.0, 6.8, 2.6, 1.6, "Goal\n$\\theta^*$ or axis", "#fef3e8"),
        (1.2, 3.6, 3.2, 1.8, "Reeds-Shepp\n$h(n)$ + analytic\nexpansion", "#edf7ed"),
        (5.2, 3.6, 3.6, 1.8, "Optimal $J^*$\n$\\rho$ on reverse arcs", "#f5f5f5"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.08,rounding_size=0.15",
                facecolor=color, edgecolor="#444", lw=1.0,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5)

    for p0, p1 in [
        ((3.0, 7.6), (3.6, 7.6)), ((6.4, 7.6), (7.0, 7.6)),
        ((8.3, 6.8), (8.3, 5.6)), ((8.3, 5.6), (4.4, 5.4)),
        ((4.4, 4.5), (5.2, 4.5)), ((2.8, 3.6), (2.8, 2.4)), ((2.8, 2.4), (6.8, 2.4)),
    ]:
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))


def draw_goal_manifold_legend(
    ax: plt.Axes,
    axis_theta: float,
    required_theta: float,
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(r"Goal set in $\theta$", fontsize=10, pad=10)

    th_axis = axis_theta
    th_opp = axis_theta + math.pi

    for col, (title, mode) in enumerate(
        [
            (r"Directed $\theta^*$", "directed"),
            (r"Axis $\{\theta^*, \theta^*\!+\!\pi\}$", "quotient"),
        ]
    ):
        ax.text(0.25 + col * 0.5, 0.96, title, ha="center", fontsize=7.5, fontweight="bold")
        inset = ax.inset_axes([0.06 + col * 0.48, 0.12, 0.4, 0.78])
        inset.set_aspect("equal")
        inset.set_xlim(-1.6, 1.6)
        inset.set_ylim(-1.6, 1.6)
        inset.axis("off")
        inset.plot(0, 0, "ko", ms=6, zorder=5)

        for th, lbl in [(th_axis, r"$\theta^*$"), (th_opp, r"$\theta^*\!+\!\pi$")]:
            if mode == "directed":
                valid = abs((th - required_theta + math.pi) % (2 * math.pi) - math.pi) < 0.01
            else:
                valid = True
            color = "#2980b9" if (valid and mode == "directed") else "#27ae60" if valid else "#c0392b"
            ls = "-" if valid else "--"
            alpha = 1.0 if valid else 0.45
            mark = "" if valid else " (invalid)"
            dx, dy = 0.85 * math.cos(th), 0.85 * math.sin(th)
            inset.annotate(
                "", xy=(dx, dy), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2, ls=ls, alpha=alpha),
            )
            inset.text(1.12 * dx, 1.12 * dy, lbl + mark, fontsize=6.5, ha="center", va="center", color=color)

        if mode == "quotient":
            inset.plot(
                [math.cos(th_axis), -math.cos(th_axis)],
                [math.sin(th_axis), -math.sin(th_axis)],
                color="#27ae60", lw=2.5, solid_capstyle="round", zorder=2,
            )


def draw_config_table(
    ax: plt.Axes,
    mean_a: float,
    mean_b: float,
    mean_rho1: float,
) -> None:
    ax.axis("off")
    ax.set_title(r"Mean $J^*$ (500 samples)", fontsize=10, pad=12)
    rows = [
        ["A", r"directed $\theta^*$", "2", f"{mean_a:.2f}"],
        ["B", r"axis $\{\theta^*, \theta^*\!+\!\pi\}$", "1", f"{mean_b:.2f}"],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["", r"Goal $\theta$", r"$\rho$", r"$\langle J^* \rangle$"],
        loc="center",
        cellLoc="center",
        colColours=["#ddd"] * 4,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")
        if c == 0 and r > 0:
            cell.set_facecolor(["#d6e4ff", "#d5f5e3"][r - 1])
    ax.text(
        0.5, 0.04,
        r"$J^*(B) \leq J^*(A)$",
        transform=ax.transAxes, ha="center", fontsize=8, color="#333",
    )
    ax.text(
        0.5, -0.04,
        rf"symmetric-$\rho$ baseline $\langle J^* \rangle = {mean_rho1:.2f}$ (decomposition only)",
        transform=ax.transAxes, ha="center", fontsize=7, color="#666",
    )


def draw_vehicle(
    ax: plt.Axes,
    x: float,
    y: float,
    theta: float,
    *,
    symmetric: bool = False,
    color: str = "black",
    alpha: float = 1.0,
) -> None:
    length, width = 0.9, 0.45
    corners = np.array(
        [
            [length / 2, width / 2], [length / 2, -width / 2],
            [-length / 2, -width / 2], [-length / 2, width / 2],
            [length / 2, width / 2],
        ]
    )
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    body = corners @ rot.T + np.array([x, y])
    ax.plot(body[:, 0], body[:, 1], color=color, lw=1.5, alpha=alpha)
    if not symmetric:
        nose = np.array([[length / 2, 0], [length / 2 + 0.25, 0]]) @ rot.T + [x, y]
        ax.plot(nose[:, 0], nose[:, 1], color=color, lw=2, alpha=alpha)


def plot_path_panel(
    ax: plt.Axes,
    start: Pose,
    path,
    goal,
    *,
    title: str,
    subtitle: str,
    color: str,
    arrival_theta: float,
    symmetric_body: bool,
) -> None:
    x_cur, y_cur, th_cur = start.x, start.y, start.theta
    for elem in path:
        seg_xs, seg_ys, _, x_cur, y_cur, th_cur = _integrate_for_plot(
            x_cur, y_cur, th_cur, elem, 0.04
        )
        ls = "--" if elem.gear is Gear.BACKWARD else "-"
        col = COLOR_REVERSE if elem.gear is Gear.BACKWARD else color
        ax.plot(seg_xs, seg_ys, ls=ls, color=col, lw=2.2)

    draw_vehicle(ax, start.x, start.y, start.theta, color=color, alpha=0.85)
    draw_vehicle(ax, goal.x, goal.y, arrival_theta, symmetric=symmetric_body, color=color)
    ax.plot(goal.x, goal.y, "o", color=color, ms=5, zorder=6)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
    ax.text(
        0.03, 0.97, subtitle,
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#ccc"),
    )


def draw_decomposition_scatter(
    ax: plt.Axes,
    delta_rho: np.ndarray,
    delta_theta: np.ndarray,
    delta_total: np.ndarray,
) -> None:
    sc = ax.scatter(
        delta_rho, delta_theta, c=delta_total, cmap="viridis", s=14, alpha=0.7, edgecolors="none"
    )
    plt.colorbar(sc, ax=ax, label=CBAR_TOTAL, fraction=0.046, pad=0.04)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel(XLABEL_RHO)
    ax.set_ylabel(YLABEL_GOAL)
    ax.set_title(r"Cost decomposition ($n=500$)", fontsize=10, pad=10)
    ax.grid(True)


def make_overview_figure(
    *,
    start: Pose,
    goal_a: DirectedGoal,
    goal_b: QuotientGoal,
    path_a,
    path_b,
    arrival_theta_b: float,
    data: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    use_publication_style()
    cost_a = path_length(path_a, 2.0)
    cost_b_val = path_length(path_b, 1.0)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(OVERVIEW_TITLE, fontsize=13, fontweight="bold", y=0.97)
    gs = fig.add_gridspec(
        4, 3,
        height_ratios=[0.95, 1.35, 1.05, 0.85],
        hspace=0.55,
        wspace=0.35,
        left=0.07,
        right=0.97,
        top=0.91,
        bottom=0.07,
    )

    draw_stack_pipeline(fig.add_subplot(gs[0, :]))
    draw_goal_manifold_legend(
        fig.add_subplot(gs[1, 0]),
        axis_theta=goal_b.theta,
        required_theta=goal_a.theta,
    )

    ax_a = fig.add_subplot(gs[1, 1])
    plot_path_panel(
        ax_a, start, path_a, goal_a,
        title=TITLE_A,
        subtitle=rf"$J^* = {cost_a:.2f}$",
        color=COLOR_A,
        arrival_theta=goal_a.theta,
        symmetric_body=False,
    )

    ax_b = fig.add_subplot(gs[1, 2])
    plot_path_panel(
        ax_b, start, path_b, goal_b,
        title=TITLE_B,
        subtitle=rf"$J^* = {cost_b_val:.2f}$",
        color=COLOR_B,
        arrival_theta=arrival_theta_b,
        symmetric_body=True,
    )
    for ax in (ax_a, ax_b):
        ax.set_xlim(-4.8, 1.2)
        ax.set_ylim(-1.8, 1.8)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")

    draw_config_table(
        fig.add_subplot(gs[2, 0]),
        data["ja"].mean(),
        data["jb"].mean(),
        data["j_rho1"].mean(),
    )
    draw_decomposition_scatter(
        fig.add_subplot(gs[2, 1:]),
        data["delta_rho"], data["delta_theta"], data["delta_total"],
    )

    draw_figure_legend(fig.add_subplot(gs[3, :]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
