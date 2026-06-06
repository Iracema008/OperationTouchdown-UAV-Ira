''' Lawnmower path planning'''

# planner="grid"  → plain  order, no optimzation, no replan
# planner="sa"    → Simulated annealing optimized order (pre-flight) &  mid-flight replan
# this is the old confirm position version 

import math
import time
import threading
import numpy as np
from typing import Optional
from pymavlink import mavutil

from core.log import get_logger
from landing.pixhawk_controller.stationary_landing_controller import StationaryLandingController
from path_planning.simulated_annealing import build_sa_waypoints, replan_remaining

logger = get_logger(__name__)


FIELD_CONFIG = {
    "north_min_m": 0.0,
    "north_max_m": 9.2,
    "east_min_m": 0.0,
    "east_max_m": 9.2,
    "search_alt_m": 3.0,
    "confirm_alt_m": 1.2,
    "wp_accept_radius_m": 0.4,
    "wp_timeout_s": 20.0,
    "move_speed_ms": 1.2, # maybe 3? 
}

LAWNMOWER_CONFIG = {
    # Derived from OAK-D S2 real intrinsics at 640x480, 3m altitude, 10% overlap
    # TODO: need to switch this to 640 by 400 .
    "col_spacing_m": 2.8,
    "row_spacing_m": 2.5,
}



def dist2d(a: tuple, b: tuple) -> float:
    """Euclidean distance between two (north, east) points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def vo_north_east(vo) -> tuple:
    """
    Return current (north, east) from VO object.
    VO pos layout: [x_right, y_down, z_forward]
    """
    pos, _ = vo.pose()
    return float(pos[2]), float(pos[0])


def send_goto_ned(master, north: float, east: float, down: float):
    """Send LOCAL_NED position setpoint to Pixhawk."""
    type_mask = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
    )
    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        north, east, down,
        0, 0, 0,
        0, 0, 0,
        0, 0,
    )


def build_lawnmower_waypoints(cfg: dict, lm_cfg: dict) -> list:
    """ Generates grid waypoints, returns list of (north, east) tuples in raw sweep order. """
    waypoints   = []
    col_spacing = lm_cfg["col_spacing_m"]
    row_spacing = lm_cfg["row_spacing_m"]

    north_min = cfg["north_min_m"]
    north_max = cfg["north_max_m"]
    east_min  = cfg["east_min_m"]
    east_max  = cfg["east_max_m"]

    east    = east_min
    col_idx = 0

    while east <= east_max + 1e-6:
        if col_idx % 2 == 0:
            north = north_min
            while north <= north_max + 1e-6:
                waypoints.append((north, east))
                north += row_spacing
        else:
            north = north_max
            while north >= north_min - 1e-6:
                waypoints.append((north, east))
                north -= row_spacing
        east += col_spacing
        col_idx += 1

    logger.info(
        f"[Lawnmower] Generated {len(waypoints)} grid waypoints "
        f"({col_idx} columns)"
    )
    return waypoints


def navigate_to(master, vo, cfg: dict, north: float, east: float, down: float, label: str, marker_confirmed, stop_event: threading.Event) -> bool:
    """
    Fly to NED position, block until arrival or timeout.

    Polls marker_confirmed and stop_event every 100ms — returns False
    immediately if either fires so the sweep aborts with no delay.

    Returns
    -------
    True  : arrived within wp_accept_radius_m
    False : timed out OR marker_confirmed set OR stop_event set
    """
    t_start  = time.time()
    timeout  = cfg["wp_timeout_s"]
    r_accept = cfg["wp_accept_radius_m"]

    while not stop_event.is_set():
        send_goto_ned(master, north, east, down)

        cur = vo_north_east(vo)
        if dist2d(cur, (north, east)) <= r_accept:
            return True

        if time.time() - t_start > timeout:
            logger.warning(
                f"[Lawnmower] WP timeout {label} N={north:.1f} E={east:.1f}"
            )
            return False

        if marker_confirmed.is_set():
            logger.info(
                f"[Lawnmower] marker_confirmed mid-transit — aborting {label}"
            )
            return False

        time.sleep(0.1)

    return False


def fly_to_marker_and_land(master, vo, cfg: dict,
                            controller: StationaryLandingController,
                            north: float, east: float,
                            marker_confirmed,
                            stop_event: threading.Event,
                            on_confirmed,
                            mission_state: dict):
    """
    Descend to confirm altitude over marker VO position, then land.

    NOTE: When re-enabling AprilTag precision landing, comment out
    stationary_landing() + disarm_motors() and let run_landing take over.
    """
    confirm_down = -cfg["confirm_alt_m"]

    logger.info(
        f"[Lawnmower] Descending to confirmed marker — "
        f"N={north:.2f} E={east:.2f} at {cfg['confirm_alt_m']}m"
    )

    navigate_to(
        master, vo, cfg,
        north, east, confirm_down,
        label="marker-approach",
        marker_confirmed=marker_confirmed,
        stop_event=stop_event,
    )

    if stop_event.is_set():
        return

    final_north, final_east = vo_north_east(vo)
    logger.info(
        f"[Lawnmower] At confirm altitude — VO N={final_north:.2f} E={final_east:.2f}"
    )

    mission_state["valid_marker_confirmed"]    = True
    mission_state["confirmed_marker_position"] = (final_north, final_east)

    if on_confirmed:
        on_confirmed(final_north, final_east)

    # TODO: comment these two lines out when re-enabling AprilTag landing
    logger.info("[Lawnmower] Calling stationary_landing()")
    controller.stationary_landing()
    controller.disarm_motors()

    stop_event.set()


def run_flight_loop(master, vo, cfg: dict, waypoints: list,  controller: StationaryLandingController,
                    marker_confirmed, uncertain_pos, stop_event: threading.Event, on_confirmed,
                    mission_state: dict, planner: str = "grid"):
    """
    Main lawnmower flight loop.

    planner="grid":
        Fly waypoints in given order. uncertain_pos is ignored — no replan.

    planner="sa":
        Same flight, but when uncertain_pos[2] == 1.0 (vision saw valid ID
        2 consecutive times) the remaining unvisited waypoints are
        reordered ONCE via simulated_annealing.replan_remaining().

    marker_confirmed (multiprocessing.Event) is the authoritative landing
    signal regardless of planner — it aborts the sweep immediately.

    uncertain_pos: multiprocessing.Array('d', [north, east, flag])
    """
    search_down = -cfg["search_alt_m"]
    total       = len(waypoints)
    replan_done = False
    use_sa      = (planner == "sa")

    idx = 0
    while idx < len(waypoints):
        if stop_event.is_set():
            break

        if marker_confirmed.is_set():
            logger.info(
                f"[Lawnmower] marker_confirmed before WP {idx + 1} — aborting"
            )
            break

        # only in simulated annealing mode
        if use_sa and not replan_done and uncertain_pos[2] == 1.0:
            detection_pos = (uncertain_pos[0], uncertain_pos[1])
            current_pos   = vo_north_east(vo)
            remaining     = waypoints[idx:]

            logger.info(
                f"[Lawnmower] SA replan triggered — detection "
                f"N={detection_pos[0]:.2f} E={detection_pos[1]:.2f}, "
                f"{len(remaining)} waypoints remaining"
            )

            replanned   = replan_remaining(remaining, current_pos, detection_pos)
            waypoints   = waypoints[:idx] + replanned
            total       = len(waypoints)
            replan_done = True

        north, east = waypoints[idx]

        logger.info(
            f"[Lawnmower] WP {idx + 1}/{total} -> N={north:.1f} E={east:.1f}"
            f"{' [SA]' if replan_done else ''}"
        )

        navigate_to(
            master, vo, cfg,
            north, east, search_down,
            label=f"wp{idx + 1}",
            marker_confirmed=marker_confirmed,
            stop_event=stop_event,
        )

        if marker_confirmed.is_set():
            break

        idx += 1

    # Marker confirmed during sweep, fly to it and land
    if marker_confirmed.is_set() and not stop_event.is_set():
        snap_north, snap_east = vo_north_east(vo)
        logger.info(
            f"[Lawnmower] Marker confirmed — VO snapshot "
            f"N={snap_north:.2f} E={snap_east:.2f}"
        )
        fly_to_marker_and_land(
            master, vo, cfg,
            controller=controller,
            north=snap_north,
            east=snap_east,
            marker_confirmed=marker_confirmed,
            stop_event=stop_event,
            on_confirmed=on_confirmed,
            mission_state=mission_state,
        )

    if not mission_state["valid_marker_confirmed"]:
        logger.info("[Lawnmower] Sweep complete — valid marker not found")
        stop_event.set()


def start_lawnmower_search(mav_master, vo, valid_ids: list,  marker_confirmed, uncertain_pos, controller: StationaryLandingController,
                            planner: str = "grid",  field_cfg: dict = None, lm_cfg: dict = None,  on_confirmed=None) -> dict:
    """
    Returns mission_state dict the caller can poll.
    """
    cfg = field_cfg or FIELD_CONFIG
    lm  = lm_cfg   or LAWNMOWER_CONFIG

    # Always build the base grid first
    grid = build_lawnmower_waypoints(cfg, lm)

    # Planner selection — this is the ONLY branch that differs between modes
    if planner == "sa":
        waypoints = build_sa_waypoints(grid, start_pos=(0.0, 0.0))
        logger.info("[Lawnmower] Planner = SA (optimised order + replan)")
    else:
        waypoints = grid
        logger.info("[Lawnmower] Planner = GRID (plain boustrophedon)")

    stop_event = threading.Event()

    mission_state = {
        "valid_marker_confirmed":    False,
        "confirmed_marker_position": None,
        "stop_event":                stop_event,
        "planner":                   planner,
    }

    flight_thread = threading.Thread(
        target=run_flight_loop,
        args=(
            mav_master, vo, cfg,
            waypoints,
            controller,
            marker_confirmed,
            uncertain_pos,
            stop_event,
            on_confirmed,
            mission_state,
            planner,
        ),
        daemon=True,
        name="LawnmowerFlight",
    )
    flight_thread.start()

    logger.info(
        f"[Lawnmower] Search started — {len(waypoints)} waypoints, "
        f"planner={planner}, valid IDs={list(valid_ids)}"
    )
    return mission_state


def run_lawnmower_mission(mav_master, vo,  valid_ids: list, marker_confirmed, uncertain_pos, controller: StationaryLandingController,
                           planner: str = "grid", field_cfg: dict = None, lm_cfg: dict = None,  on_confirmed=None) -> dict:
    """ Blocking version of start_lawnmower_search, for planner : "grid" or "sa"
     Blocks until marker confirmed and landed, or field exhausted.
    """
    mission_state = start_lawnmower_search(
        mav_master, vo,
        valid_ids=valid_ids,
        marker_confirmed=marker_confirmed,
        uncertain_pos=uncertain_pos,
        controller=controller,
        planner=planner,
        field_cfg=field_cfg,
        lm_cfg=lm_cfg,
        on_confirmed=on_confirmed,
    )

    stop_event = mission_state["stop_event"]

    try:
        while not mission_state["valid_marker_confirmed"]:
            if stop_event.is_set():
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.warning("[LawnmowerRunner] Interrupted by user")
        stop_event.set()
        return mission_state

    if mission_state["valid_marker_confirmed"]:
        pos = mission_state["confirmed_marker_position"]
        logger.info(
            f"[LawnmowerRunner] Mission complete ({planner}) — "
            f"landed at N={pos[0]:.2f} E={pos[1]:.2f}"
        )
    else:
        logger.warning(
            f"[LawnmowerRunner] Mission ended ({planner}) — marker not found"
        )

    return mission_state