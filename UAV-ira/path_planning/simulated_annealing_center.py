''' Simulated Annealing with center-biased cost function.

Same SA algorithm as simulated_annealing.py but the cost function adds
a penalty for visiting waypoints far from the field center early in the
path. This pulls center waypoints toward the front of the search order
without changing the waypoints themselves.

    --planner sac  → center-biased SA (this file)
    --planner sa   → plain distance SA (simulated_annealing.py)
    --planner grid → boustrophedon grid (mission_grid.py)

Why center bias helps:
    Competition judges place the ArUco marker randomly on the field.
    Statistically the center is more likely than any specific edge or corner.
    Visiting the center first reduces expected discovery time without
    sacrificing full field coverage — the drone still visits every waypoint,
    just in a center-first order.

Tuning PENALTY_STRENGTH:
    0.0  → identical to plain SA (no bias)
    0.5  → mild bias, mostly distance-focused
    1.0  → balanced distance and center bias (default)
    2.0  → strong bias, willing to fly extra to visit center early
    If the optimized path looks like it backtracks badly, lower this value.
    If it looks identical to grid, raise it.
'''

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

# Controls how strongly center proximity is rewarded.
# Increase if the path still looks like plain grid.
# Decrease if the path backtracks excessively.
PENALTY_STRENGTH = 3.0


def dist2d(a: tuple, b: tuple) -> float:
    '''Euclidean distance between two (north, east) points.'''
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def total_path_distance(waypoints: list, start: tuple = (0.0, 0.0)) -> float:
    '''Total path length: start → wp0 → wp1 → ... → wpN.'''
    if not waypoints:
        return 0.0
    dist = dist2d(start, waypoints[0])
    for i in range(len(waypoints) - 1):
        dist += dist2d(waypoints[i], waypoints[i + 1])
    return dist


def center_penalty(waypoints: list, field_center: tuple, penalty_strength: float) -> float:
    '''Penalize orderings that visit waypoints far from center early.

    Each waypoint contributes: position_weight × dist_from_center × penalty_strength
    position_weight is 1.0 for the first waypoint and 0.0 for the last.
    So center waypoints visited early produce low penalty.
    Edge waypoints visited early produce high penalty.
    '''
    if not waypoints or penalty_strength == 0.0:
        return 0.0
    n       = len(waypoints)
    penalty = 0.0
    for i, wp in enumerate(waypoints):
        position_weight = 1.0 - (i / n)   # 1.0 for first WP → 0.0 for last WP
        penalty += position_weight * dist2d(wp, field_center)
    return penalty * penalty_strength


def total_cost(waypoints: list, start: tuple, field_center: tuple,
               penalty_strength: float) -> float:
    '''Combined cost = path distance + center penalty.
    This is what center-biased SA minimises instead of distance alone.
    '''
    return (
        total_path_distance(waypoints, start) +
        center_penalty(waypoints, field_center, penalty_strength)
    )


def run_sa_center(waypoints: list, start_pos: tuple, field_center: tuple,
                  penalty_strength: float = PENALTY_STRENGTH,
                  sa_cfg: dict = None) -> list:
    '''SA with center-biased cost function.

    Same swap-and-accept algorithm as run_sa() but evaluates candidates
    using total_cost() instead of total_path_distance(). The center penalty
    causes SA to prefer orderings that visit center waypoints early.
    '''
    if len(waypoints) <= 2:
        return waypoints

    cfg      = sa_cfg or SA_CONFIG
    temp     = cfg["initial_temp"]
    cool     = cfg["cooling_rate"]
    min_temp = cfg["min_temp"]
    iters    = cfg["iterations_per_temp"]

    current   = list(waypoints)
    best      = list(current)
    cur_cost  = total_cost(current, start_pos, field_center, penalty_strength)
    best_cost = cur_cost
    start_dist = total_path_distance(current, start_pos)
    steps = 0

    while temp > min_temp:
        for _ in range(iters):
            i, j      = random.sample(range(len(current)), 2)
            candidate = list(current)
            candidate[i], candidate[j] = candidate[j], candidate[i]

            cand_cost = total_cost(candidate, start_pos, field_center, penalty_strength)
            delta     = cand_cost - cur_cost

            # Always accept improvements, sometimes accept worse to escape local minima
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current  = candidate
                cur_cost = cand_cost
                if cur_cost < best_cost:
                    best      = list(current)
                    best_cost = cur_cost

        temp  *= cool
        steps += 1

    end_dist = total_path_distance(best, start_pos)
    logger.info(
        f"[SAC] {len(waypoints)} waypoints — "
        f"dist {start_dist:.2f}m → {end_dist:.2f}m | "
        f"penalty_strength={penalty_strength} | "
        f"{steps} steps"
    )
    return best


def build_sa_center_waypoints(grid_waypoints: list, start_pos: tuple = (0.0, 0.0),
                               north_max: float = 8.0, east_max: float = 8.0,
                               penalty_strength: float = PENALTY_STRENGTH,
                               sa_cfg: dict = None) -> list:
    '''Pre-flight center-biased optimisation.

    Takes the raw lawnmower grid and returns a waypoint order that visits
    center waypoints earlier than edge waypoints while still covering the
    full field. north_max and east_max must match FIELD_CONFIG in
    mission_sa_center.py so the field center is computed correctly.
    '''
    field_center = (north_max / 2.0, east_max / 2.0)
    logger.info(
        f"[SAC] Running center-biased pre-flight optimisation — "
        f"field center N={field_center[0]:.1f} E={field_center[1]:.1f} | "
        f"penalty_strength={penalty_strength}"
    )
    return run_sa_center(
        grid_waypoints, start_pos, field_center, penalty_strength, sa_cfg
    )