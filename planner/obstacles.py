from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

VEHICLE_LENGTH = 0.9
VEHICLE_WIDTH = 0.45
VEHICLE_CLEARANCE = 0.08


@dataclass(frozen=True)
class RectObstacle:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin


@dataclass(frozen=True)
class ObstacleMap:
    rects: tuple[RectObstacle, ...] = ()

    def vehicle_corners(self, x: float, y: float, theta: float) -> np.ndarray:
        half_l = VEHICLE_LENGTH / 2
        half_w = VEHICLE_WIDTH / 2
        local = np.array(
            [
                [half_l, half_w],
                [half_l, -half_w],
                [-half_l, -half_w],
                [-half_l, half_w],
            ]
        )
        c, s = math.cos(theta), math.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + np.array([x, y])

    def _expanded(self, rect: RectObstacle, clearance: float) -> RectObstacle:
        return RectObstacle(
            rect.xmin - clearance,
            rect.ymin - clearance,
            rect.xmax + clearance,
            rect.ymax + clearance,
        )

    def _point_in_rect(self, px: float, py: float, rect: RectObstacle) -> bool:
        return rect.xmin <= px <= rect.xmax and rect.ymin <= py <= rect.ymax

    def _segment_intersects_rect(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        rect: RectObstacle,
    ) -> bool:
        if self._point_in_rect(x0, y0, rect) or self._point_in_rect(x1, y1, rect):
            return True
        edges = (
            ((rect.xmin, rect.ymin), (rect.xmax, rect.ymin)),
            ((rect.xmax, rect.ymin), (rect.xmax, rect.ymax)),
            ((rect.xmax, rect.ymax), (rect.xmin, rect.ymax)),
            ((rect.xmin, rect.ymax), (rect.xmin, rect.ymin)),
        )

        def cross(ax, ay, bx, by, cx, cy) -> float:
            return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

        for (ex0, ey0), (ex1, ey1) in edges:
            d1 = cross(ex0, ey0, ex1, ey1, x0, y0)
            d2 = cross(ex0, ey0, ex1, ey1, x1, y1)
            d3 = cross(x0, y0, x1, y1, ex0, ey0)
            d4 = cross(x0, y0, x1, y1, ex1, ey1)
            if d1 * d2 <= 0 and d3 * d4 <= 0:
                return True
        return False

    def _obb_hits_rect(self, corners: np.ndarray, rect: RectObstacle) -> bool:
        for px, py in corners:
            if self._point_in_rect(float(px), float(py), rect):
                return True
        for i in range(4):
            j = (i + 1) % 4
            if self._segment_intersects_rect(
                float(corners[i, 0]),
                float(corners[i, 1]),
                float(corners[j, 0]),
                float(corners[j, 1]),
                rect,
            ):
                return True
        cx = float(np.mean(corners[:, 0]))
        cy = float(np.mean(corners[:, 1]))
        if self._point_in_rect(cx, cy, rect):
            return True
        return False

    def vehicle_collision(
        self,
        x: float,
        y: float,
        theta: float,
        *,
        clearance: float = VEHICLE_CLEARANCE,
    ) -> bool:
        corners = self.vehicle_corners(x, y, theta)
        for rect in self.rects:
            if self._obb_hits_rect(corners, self._expanded(rect, clearance)):
                return True
        return False

    def collision(self, x: float, y: float, theta: float = 0.0) -> bool:
        return self.vehicle_collision(x, y, theta)

    def draw(self, ax: plt.Axes) -> list[Rectangle]:
        patches: list[Rectangle] = []
        for rect in self.rects:
            patch = Rectangle(
                (rect.xmin, rect.ymin),
                rect.width,
                rect.height,
                facecolor="#cbd5e1",
                edgecolor="#64748b",
                linewidth=1.0,
                alpha=0.92,
                zorder=2,
            )
            ax.add_patch(patch)
            patches.append(patch)
        return patches

    def bounds(self) -> tuple[float, float, float, float] | None:
        if not self.rects:
            return None
        xs = [r.xmin for r in self.rects] + [r.xmax for r in self.rects]
        ys = [r.ymin for r in self.rects] + [r.ymax for r in self.rects]
        return min(xs), max(xs), min(ys), max(ys)
