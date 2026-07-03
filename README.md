# Bidirectional motion planning

## What this is

This repo compares two ways to plan paths for a car that can drive forward and reverse equally (Reeds-Shepp / bicycle model):

- Vehicle A (conventional): must arrive at the goal facing a specific direction θ, and pays extra cost for every reverse segment (ρ = 2).
- Vehicle B (bidirectional): may arrive facing either direction along the same axis (θ or θ + π), with no reverse penalty markup (ρ = 1).

The repo plans those paths, measures how much cheaper B is than A, and splits those benefits into two effects: reverse-penalty symmetry vs relaxing the goal heading.

Outputs:


| Command                                | Produces                                                                          |
| -------------------------------------- | --------------------------------------------------------------------------------- |
| `uv run python experiments/compare.py` | `output/overview.png`: 500 random free-space trials, cost decomposition           |
| `uv run python experiments/animate.py` | `output/compare.mp4`: 10 hand-designed scenes with obstacles, A vs B side by side |




## How planning works (end to end)



### 1. Pick a goal type

Every plan starts with a pose `(x, y, heading)` and a goal. The only difference between vehicle A and B is what "arriving at the goal" means:

- A, directed goal: position and heading must both match (e.g. "park here, nose pointing left").
- B, axis goal: position must match, but heading can be either way along the same axis (e.g. "park here, nose left or right is fine").

That choice lives in `models/symmetry.py` (`DirectedGoal` vs `QuotientGoal`).

### 2. Reeds-Shepp: optimal paths in open space (not implemented here)

When nothing is in the way, the shortest car path between two poses is a Reeds-Shepp curve, a sequence of straight segments and turns, possibly including reverse.

This repo does not implement those curves. It calls the Rust `[reeds_shepp](https://linusweigand.github.io/reeds-shepp/#get-started)` crate through thin Python bindings (`planner/reeds_shepp.py`, `src/lib.rs`). Path sampling and cost helpers live in Python; path enumeration requires the Rust extension built by `uv sync`.

Reeds-Shepp is used in three places:

1. Direct planning when a scenario has no obstacles (fast, exact optimum in free space).
2. Heuristic during search: "how far am I from the goal in open space?" guides which nodes hybrid A* expands first.
3. Shortcut / handoff: whenever the search gets close enough to the goal, it tries to connect the current pose to the goal with a full Reeds-Shepp path. If that path is collision-free, planning stops and the final trajectory is that analytic curve (possibly prefixed by a short lattice segment to reach the handoff point).



### 3. Hybrid A*: search when obstacles are in the way

Obstacle scenes use hybrid A* (`planner/hybrid_astar.py`), the standard pattern from the DARPA Urban Challenge:

1. Discretize position and heading into a 3D grid.
2. Expand the start node with short motion primitives: small forward/reverse arcs at a few steering angles (bicycle model, minimum turning radius 1 m).
3. Reject any step where the full vehicle rectangle hits an obstacle (`planner/obstacles.py` checks the car footprint, not just its center).
4. Score each node with cost-so-far plus the Reeds-Shepp heuristic to the goal.
5. Try analytic handoff on every expanded node within a radius of the goal: compute the best Reeds-Shepp path to the goal, collision-check it densely, and if it clears, return that path immediately; no need to search all the way to the goal cell by cell.
6. Replay the chosen motion primitives (or lattice prefix + Reeds-Shepp tail) into a dense `(x, y, θ)` trajectory for plotting and animation.

So hybrid A* is the obstacle-avoiding search; Reeds-Shepp is the "finish line" solver and the estimate of how far remains. In open space, step 5 succeeds on the first try and you never really need the lattice.

### 4. Cost and the A vs B comparison

Path cost is total arc length, with each reverse segment multiplied by ρ (2 for A, 1 for B).

On 500 random free-space trials, B's optimal cost is never worse than A's, and usually better. The gap splits cleanly:


| Term | Meaning                                                                        |
| ---- | ------------------------------------------------------------------------------ |
| Δρ   | How much A overpays because ρ = 2 penalizes reverse while B uses ρ = 1         |
| Δθ   | How much B saves by allowing either heading at the goal instead of one fixed θ |


Roughly: cost(A) − cost(B) ≈ Δρ + Δθ. The experiment figure labels these on each trial; the video shows the same split scene by scene.

## Repository layout

```
models/symmetry.py           # Directed vs axis goal definitions
planner/hybrid_astar.py      # Lattice search + Reeds-Shepp handoff (this repo)
planner/obstacles.py         # Vehicle footprint collision + drawing
planner/reeds_shepp.py       # Python API + path sampling; enumeration via Rust
src/lib.rs                   # PyO3 bindings
experiments/compare.py       # 500-trial figure → output/overview.png
experiments/animate.py       # 10-scenario video → output/compare.mp4
experiments/viz.py           # Shared figure styling
```



## Running it

Requires [uv](https://docs.astral.sh/uv/), Rust (`rustup`, to build the Reeds-Shepp extension), and ffmpeg (for the MP4).

```bash
uv sync
uv run python experiments/compare.py
uv run python experiments/animate.py   # → output/compare.mp4
```

Run from the repo root. Generated files land in `output/` (gitignored except `.gitkeep`).

Quick API example (free space, axis goal):

```python
from planner.hybrid_astar import HybridAStar, make_goal
from planner.reeds_shepp import Pose

start = Pose(0, 0, 0)
goal = make_goal(5, 3, 1.2, quotient=True, reverse_penalty=1.0)
planner = HybridAStar(goal=goal, reverse_penalty=1.0)
result = planner.plan(start)   # Reeds-Shepp handoff on first expansion
```

With obstacles, pass a collision checker:

```python
from planner.obstacles import ObstacleMap

obstacles = ObstacleMap((...))
planner = HybridAStar(
    goal=goal,
    reverse_penalty=1.0,
    collision_checker=obstacles.vehicle_collision,
)
```

