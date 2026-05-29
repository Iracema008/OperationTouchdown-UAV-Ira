''' Simulated Annealing Search Pattern '''

"""

    --planner grid  → plain boustrophedon order, no SA (lawnmower.py)
    --planner sa    → SA-optimised order + mid-flight replan (this file)

Waypoint format (shared contract with lawnmower.py):
    list of (north, east) float tuples

TWO ENTRY POINTS
----------------
build_sa_waypoints(grid_waypoints, start_pos)
    Pre-flight — reorders the full grid to minimise total path distance.

replan_remaining(remaining, current_pos, detection_pos)
    Mid-flight — reorders unvisited waypoints, biased toward a location
    where vision saw the valid marker but hasn't fully confirmed it.
"""

import math
import random

from core.log import get_logger

logger = get_logger(__name__)


SA_CONFIG = {
    "initial_temp":        100.0,
    "cooling_rate":        0.95,
    "min_temp":            0.01,
    "iterations_per_temp": 10,
}


def dist2d(a: tuple, b: tuple) -> float:
    """Euclidean distance between two (north, east) points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def total_path_distance(waypoints: list, start: tuple = (0.0, 0.0)) -> float:
    """
    Total path length: start → wp0 → wp1 → ... → wpN.
    This is what SA minimises.
    """
    if not waypoints:
        return 0.0
    dist = dist2d(start, waypoints[0])
    for i in range(len(waypoints) - 1):
        dist += dist2d(waypoints[i], waypoints[i + 1])
    return dist

def run_sa(waypoints: list, start_pos: tuple, sa_cfg: dict = None) -> list:
    """
    Simulated Annealing — reorder waypoints to minimise total path distance.

    Algorithm:
        1. Begin with the given waypoint order
        2. At each temperature, randomly swap two waypoints
        3. Always accept improvements
        4. Accept worse swaps with probability exp(-delta/temp) — lets the
           search escape local minima while temperature is high
        5. Cool temperature each step until below min_temp

    """

    if len(waypoints) <= 2:
        return waypoints  # nothing meaningful to optimise

    cfg      = sa_cfg or SA_CONFIG
    temp     = cfg["initial_temp"]
    cool     = cfg["cooling_rate"]
    min_temp = cfg["min_temp"]
    iters    = cfg["iterations_per_temp"]

    current   = list(waypoints)
    best       = list(current)
    cur_dist   = total_path_distance(current, start_pos)
    best_dist  = cur_dist
    start_dist = cur_dist

    steps = 0
    while temp > min_temp:
        for _ in range(iters):
            i, j = random.sample(range(len(current)), 2)
            candidate = list(current)
            candidate[i], candidate[j] = candidate[j], candidate[i]

            cand_dist = total_path_distance(candidate, start_pos)
            delta     = cand_dist - cur_dist

            if delta < 0 or random.random() < math.exp(-delta / temp):
                current  = candidate
                cur_dist = cand_dist
                if cur_dist < best_dist:
                    best      = list(current)
                    best_dist = cur_dist

        temp *= cool
        steps += 1

    logger.info(
        f"[SA] {len(waypoints)} waypoints — "
        f"{start_dist:.2f}m → {best_dist:.2f}m over {steps} steps"
    )
    return best


def build_sa_waypoints(grid_waypoints: list,  start_pos: tuple = (0.0, 0.0), sa_cfg: dict = None) -> list:
    """ Pre-flight optimisation. Takes the raw lawnmower grid and returns  an SA-optimised visitation order.     """
    logger.info("[SA] Running pre-flight path optimisation...")
    return run_sa(grid_waypoints, start_pos, sa_cfg)


def replan_remaining(remaining_waypoints: list, current_pos: tuple,  detection_pos: tuple,  sa_cfg: dict = None) -> list:
    """
    Mid-flight replan triggered by an uncertain detection.

    Reorders ONLY the unvisited remaining waypoints to minimize distance
    from current_pos, biased toward detection_pos (where vision saw the
    valid marker but hasn't fully confirmed it yet).

    Bias technique: detection_pos is inserted as a phantom waypoint so SA
    routes toward it, then removed from the final result. This pulls the
    ordering toward the detection area without forcing an exact visit.

    Reordered remaining waypoints — detection area prioritised.
    """
    if not remaining_waypoints:
        return remaining_waypoints

    logger.info(
        f"[SA] Replanning {len(remaining_waypoints)} remaining waypoints — "
        f"detection N={detection_pos[0]:.2f} E={detection_pos[1]:.2f}"
    )

    augmented = [detection_pos] + list(remaining_waypoints)
    optimised = run_sa(augmented, current_pos, sa_cfg)

    # Strip the phantom detection waypoint back out
    result = [wp for wp in optimised if wp != detection_pos]

    logger.info(f"[SA] Replan complete — new order: {result}")
    return result