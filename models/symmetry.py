from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from planner.reeds_shepp import Pose, get_optimal_path, path_length, shortest_path_length


@dataclass(frozen=True)
class AnalyticResult:
    cost: float
    path: tuple
    goal_theta: float


class GoalManifold(ABC):
    def __init__(
        self,
        x: float,
        y: float,
        theta: float,
        *,
        turning_radius: float = 1.0,
        reverse_penalty: float = 1.0,
        pos_tol: float = 0.15,
        theta_tol: float = 0.15,
    ) -> None:
        self.x = x
        self.y = y
        self.theta = theta
        self.turning_radius = turning_radius
        self.reverse_penalty = reverse_penalty
        self.pos_tol = pos_tol
        self.theta_tol = theta_tol

    @abstractmethod
    def is_goal(self, x: float, y: float, theta: float) -> bool:
        ...

    @abstractmethod
    def heuristic(self, x: float, y: float, theta: float) -> float:
        ...

    @abstractmethod
    def analytic_expansion(
        self, x: float, y: float, theta: float
    ) -> AnalyticResult | None:
        ...

    def _pose(self, x: float, y: float, theta: float) -> Pose:
        return Pose(x, y, theta)

    def _goal_pose(self, theta: float | None = None) -> Pose:
        th = self.theta if theta is None else theta
        return Pose(self.x, self.y, th)


class DirectedGoal(GoalManifold):
    def is_goal(self, x: float, y: float, theta: float) -> bool:
        pos_ok = math.hypot(x - self.x, y - self.y) <= self.pos_tol
        dtheta = abs((theta - self.theta + math.pi) % (2 * math.pi) - math.pi)
        return pos_ok and dtheta <= self.theta_tol

    def heuristic(self, x: float, y: float, theta: float) -> float:
        return shortest_path_length(
            self._pose(x, y, theta),
            self._goal_pose(),
            self.turning_radius,
            self.reverse_penalty,
        )

    def analytic_expansion(
        self, x: float, y: float, theta: float
    ) -> AnalyticResult | None:
        start = self._pose(x, y, theta)
        goal = self._goal_pose()
        path = get_optimal_path(
            start, goal, self.turning_radius, self.reverse_penalty
        )
        cost = path_length(path, self.reverse_penalty)
        return AnalyticResult(cost, path, self.theta)


class QuotientGoal(GoalManifold):
    @property
    def axis_orientations(self) -> tuple[float, float]:
        return (self.theta, self.theta + math.pi)

    def _heading_error(self, theta: float, axis: float) -> float:
        d1 = abs((theta - axis + math.pi) % (2 * math.pi) - math.pi)
        d2 = abs((theta - axis - math.pi + math.pi) % (2 * math.pi) - math.pi)
        return min(d1, d2)

    def is_goal(self, x: float, y: float, theta: float) -> bool:
        pos_ok = math.hypot(x - self.x, y - self.y) <= self.pos_tol
        if not pos_ok:
            return False
        err = min(self._heading_error(theta, a) for a in self.axis_orientations)
        return err <= self.theta_tol

    def heuristic(self, x: float, y: float, theta: float) -> float:
        start = self._pose(x, y, theta)
        costs = [
            shortest_path_length(
                start,
                self._goal_pose(th),
                self.turning_radius,
                self.reverse_penalty,
            )
            for th in self.axis_orientations
        ]
        return min(costs)

    def analytic_expansion(
        self, x: float, y: float, theta: float
    ) -> AnalyticResult | None:
        start = self._pose(x, y, theta)
        best: AnalyticResult | None = None
        for th in self.axis_orientations:
            goal = self._goal_pose(th)
            path = get_optimal_path(
                start, goal, self.turning_radius, self.reverse_penalty
            )
            cost = path_length(path, self.reverse_penalty)
            if best is None or cost < best.cost:
                best = AnalyticResult(cost, path, th)
        return best
