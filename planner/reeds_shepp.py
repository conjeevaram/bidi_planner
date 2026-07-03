from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

try:
    import reeds_shepp_rs as _rs
except ImportError as exc:
    raise ImportError(
        "reeds_shepp_rs is required. Install with `uv sync` (needs Rust)."
    ) from exc

PI = math.pi
TWO_PI = 2.0 * PI
_EPS = 1e-10


class Steering(str, Enum):
    LEFT = "L"
    RIGHT = "R"
    STRAIGHT = "S"


class Gear(str, Enum):
    FORWARD = "F"
    BACKWARD = "B"


@dataclass(frozen=True)
class PathElement:
    steering: Steering
    gear: Gear
    length: float


Path = tuple[PathElement, ...]


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    theta: float


def normalize_angle(theta: float) -> float:
    theta = theta % TWO_PI
    if theta >= PI:
        theta -= TWO_PI
    elif theta < -PI:
        theta += TWO_PI
    return theta


def path_length(path: Path, reverse_penalty: float = 1.0) -> float:
    total = 0.0
    for e in path:
        w = reverse_penalty if e.gear is Gear.BACKWARD else 1.0
        total += w * abs(e.length)
    return total


def _integrate_segment(
    x: float,
    y: float,
    theta: float,
    elem: PathElement,
    step: float,
) -> tuple[list[float], list[float], list[float], float, float, float]:
    xs, ys, thetas = [x], [y], [theta]
    remaining = abs(elem.length)
    sign = 1.0 if elem.gear is Gear.FORWARD else -1.0

    while remaining > _EPS:
        ds = min(step, remaining)
        remaining -= ds
        if elem.steering is Steering.STRAIGHT:
            x += sign * ds * math.cos(theta)
            y += sign * ds * math.sin(theta)
        elif elem.steering is Steering.LEFT:
            x += sign * (math.sin(theta + ds) - math.sin(theta))
            y += sign * (-math.cos(theta + ds) + math.cos(theta))
            theta += sign * ds
        else:
            x += sign * (-math.sin(theta - ds) + math.sin(theta))
            y += sign * (math.cos(theta - ds) - math.cos(theta))
            theta -= sign * ds
        xs.append(x)
        ys.append(y)
        thetas.append(normalize_angle(theta))
    return xs, ys, thetas, x, y, theta


def sample_path(
    start: Pose,
    path: Path,
    step: float = 0.05,
    include_start: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Gear]]:
    xs: list[float] = []
    ys: list[float] = []
    thetas: list[float] = []

    x, y, theta = start.x, start.y, start.theta
    if include_start:
        xs.append(x)
        ys.append(y)
        thetas.append(theta)

    gears: list[Gear] = []
    for elem in path:
        seg_xs, seg_ys, seg_thetas, x, y, theta = _integrate_segment(
            x, y, theta, elem, step
        )
        xs.extend(seg_xs[1:] if len(seg_xs) > 1 else [])
        ys.extend(seg_ys[1:] if len(seg_ys) > 1 else [])
        thetas.extend(seg_thetas[1:] if len(seg_thetas) > 1 else [])
        gears.append(elem.gear)

    return np.asarray(xs), np.asarray(ys), np.asarray(thetas), gears


def _parse_rust_paths(raw: list, turning_radius: float) -> list[Path]:
    paths: list[Path] = []
    for path in raw:
        elements = []
        for item in path:
            if len(item) == 3:
                s, g, length = item
                s_str = s if isinstance(s, str) else str(s)
                g_str = g if isinstance(g, str) else str(g)
                elements.append(PathElement(Steering(s_str), Gear(g_str), length))
        if elements:
            paths.append(tuple(elements))
    return paths


def get_all_paths(
    start: Pose,
    goal: Pose,
    turning_radius: float = 1.0,
) -> list[Path]:
    raw = _rs.get_all_paths_py(
        start.x,
        start.y,
        start.theta,
        goal.x,
        goal.y,
        goal.theta,
        turning_radius,
    )
    return _parse_rust_paths(raw, turning_radius)


def get_optimal_path(
    start: Pose,
    goal: Pose,
    turning_radius: float = 1.0,
    reverse_penalty: float = 1.0,
) -> Path:
    paths = get_all_paths(start, goal, turning_radius)
    if not paths:
        dx = goal.x - start.x
        dy = goal.y - start.y
        return (PathElement(Steering.STRAIGHT, Gear.FORWARD, math.hypot(dx, dy)),)
    return min(paths, key=lambda p: path_length(p, reverse_penalty))


def shortest_path_length(
    start: Pose,
    goal: Pose,
    turning_radius: float = 1.0,
    reverse_penalty: float = 1.0,
) -> float:
    return path_length(
        get_optimal_path(start, goal, turning_radius, reverse_penalty),
        reverse_penalty,
    )
