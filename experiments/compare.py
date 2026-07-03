from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.viz import make_overview_figure
from models.symmetry import DirectedGoal, QuotientGoal
from planner.reeds_shepp import Pose, get_optimal_path

TURNING_RADIUS = 1.0
N_TRIALS = 500
RNG = np.random.default_rng(42)


def random_scenario(rng: np.random.Generator) -> tuple[Pose, float, float, float]:
    start = Pose(
        float(rng.uniform(-8, 8)),
        float(rng.uniform(-8, 8)),
        float(rng.uniform(-math.pi, math.pi)),
    )
    gx = float(rng.uniform(-8, 8))
    gy = float(rng.uniform(-8, 8))
    gtheta = float(rng.uniform(-math.pi, math.pi))
    return start, gx, gy, gtheta


def optimal_cost(
    start: Pose,
    gx: float,
    gy: float,
    gtheta: float,
    *,
    reverse_penalty: float,
    quotient: bool,
) -> float:
    if quotient:
        goal = QuotientGoal(
            gx, gy, gtheta,
            turning_radius=TURNING_RADIUS,
            reverse_penalty=reverse_penalty,
        )
    else:
        goal = DirectedGoal(
            gx, gy, gtheta,
            turning_radius=TURNING_RADIUS,
            reverse_penalty=reverse_penalty,
        )
    return goal.heuristic(start.x, start.y, start.theta)


def run_trials(n: int = N_TRIALS) -> dict[str, np.ndarray]:
    delta_rho: list[float] = []
    delta_theta: list[float] = []
    delta_total: list[float] = []
    ja: list[float] = []
    jb: list[float] = []
    j_rho1: list[float] = []

    for _ in range(n):
        start, gx, gy, gtheta = random_scenario(RNG)
        cost_a = optimal_cost(
            start, gx, gy, gtheta, reverse_penalty=2.0, quotient=False
        )
        cost_rho1 = optimal_cost(
            start, gx, gy, gtheta, reverse_penalty=1.0, quotient=False
        )
        cost_b = optimal_cost(
            start, gx, gy, gtheta, reverse_penalty=1.0, quotient=True
        )
        ja.append(cost_a)
        j_rho1.append(cost_rho1)
        jb.append(cost_b)
        delta_rho.append(cost_a - cost_rho1)
        delta_theta.append(cost_rho1 - cost_b)
        delta_total.append(cost_a - cost_b)

    return {
        "ja": np.asarray(ja),
        "jb": np.asarray(jb),
        "j_rho1": np.asarray(j_rho1),
        "delta_rho": np.asarray(delta_rho),
        "delta_theta": np.asarray(delta_theta),
        "delta_total": np.asarray(delta_total),
    }


def illustrative_scenario() -> tuple[Pose, DirectedGoal, QuotientGoal]:
    start = Pose(0.0, 0.0, 0.0)
    goal_x, goal_y = -4.0, 0.0
    goal_a = DirectedGoal(
        goal_x, goal_y, math.pi,
        turning_radius=TURNING_RADIUS,
        reverse_penalty=2.0,
    )
    goal_b = QuotientGoal(
        goal_x, goal_y, 0.0,
        turning_radius=TURNING_RADIUS,
        reverse_penalty=1.0,
    )
    return start, goal_a, goal_b


def make_figure(out_path: Path) -> None:
    data = run_trials()
    start, goal_a, goal_b = illustrative_scenario()

    path_a = get_optimal_path(
        start,
        goal_a._goal_pose(),
        TURNING_RADIUS,
        reverse_penalty=2.0,
    )
    analytic_b = goal_b.analytic_expansion(start.x, start.y, start.theta)
    assert analytic_b is not None

    make_overview_figure(
        start=start,
        goal_a=goal_a,
        goal_b=goal_b,
        path_a=path_a,
        path_b=analytic_b.path,
        arrival_theta_b=analytic_b.goal_theta,
        data=data,
        out_path=out_path,
    )

    print(f"Saved figure → {out_path}")
    print(f"Mean $J^*(A)$={data['ja'].mean():.3f}, $J^*(B)$={data['jb'].mean():.3f}")
    print(f"Mean $\\Delta_\\rho$={data['delta_rho'].mean():.3f}")
    print(f"Mean $\\Delta_\\theta$={data['delta_theta'].mean():.3f}")
    print(
        "Trials with both effects > 0.01: "
        f"{np.sum((data['delta_rho'] > 0.01) & (data['delta_theta'] > 0.01))}/{N_TRIALS}"
    )


if __name__ == "__main__":
    out = ROOT / "output" / "overview.png"
    make_figure(out)
