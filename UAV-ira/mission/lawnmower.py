""" lawnmower.py search"""

import math
import time
import threading
import numpy as np
from typing import Optional
from pymavlink import mavutil

from core.log import get_logger
from mission.pixhawk_controller.stationary_landing_controller import StationaryLandingController

logger = get_logger(__name__)


FIELD_CONFIG = {
    "north_min_m":  0.0,
    "north_max_m":  5.0,   # 5m field
    "east_min_m":   0.0,
    "east_max_m":   5.0,   # 5m field
    "search_alt_m": 3.0,
    "confirm_alt_m": 1.2,
    "wp_accept_radius_m": 0.4,
    "wp_timeout_s":       20.0,
    "move_speed_ms":       1.2,
}

LAWNMOWER_CONFIG = {
    "col_spacing_m": 2.8,   # derived from real FOV at 2.5m altitude
    "row_spacing_m": 2.0,   # waypoint spacing within each column
}


def dist2d(a: tuple, b: tuple) -> float:
    """Euclidean distance between two (north, east) points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def vo_north_east(vo) -> tuple:
    """
    Return current (north, east) from a VO object.
    Assumes VO pos layout: [x_right, y_down, z_forward]
    """
    pos, _ = vo.pose()
    return float(pos[2]), float(pos[0])

def send_goto_ned(master, north: float, east: float, down: float):
    """Send a LOCAL_NED position setpoint to the Pixhawk."""
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
    """
    Generate a boustrophedon (lawnmower) waypoint list.

    Pattern:
        Column 0 (east=east_min): fly North -> South
        Column 1 (east+=spacing): fly South -> North
        Column 2 (east+=spacing): fly North -> South
        ...

    Returns list of (north, east) tuples in flight order.
    """
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
        f"[Lawnmower] Generated {len(waypoints)} waypoints "
        f"({col_idx} columns, {row_spacing}m row spacing, "
        f"{col_spacing}m column spacing)"
    )
    return waypoints

def navigate_to(master, vo, cfg: dict,
                north: float, east: float, down: float,
                label: str,
                marker_confirmed,
                stop_event: threading.Event) -> bool:
    """
    Fly to a NED position and block until arrival or timeout.

    Polls marker_confirmed (multiprocessing.Event from run_vision) and
    stop_event on every 100ms tick — returns False immediately if either
    fires so the sweep can be abandoned with no delay.

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
                f"[Lawnmower] WP timeout {label} "
                f"N={north:.1f} E={east:.1f}"
            )
            return False

        if marker_confirmed.is_set():
            logger.info(
                f"[Lawnmower] marker_confirmed mid-transit — "
                f"aborting leg to {label}"
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
    Descend to confirm altitude over the VO-snapshotted marker position,
    then call StationaryLandingController.stationary_landing().

    Parameters
    ----------
    master      : pymavlink connection (used for NED setpoints during descent)
    vo          : VO adapter (.pose())
    cfg         : FIELD_CONFIG dict
    controller  : StationaryLandingController — used for the final land command
    north/east  : VO position snapshotted when marker_confirmed fired
    marker_confirmed : multiprocessing.Event
    stop_event  : threading.Event (internal abort)
    on_confirmed: optional callback(north, east) for logging/telemetry
    mission_state: dict updated in place on success

    NOTE: When re-enabling AprilTag precision landing, comment out the
    stationary_landing() call below and instead let run_landing take over
    once marker_confirmed is set. The lawnmower would just descend and hold.
    """
    confirm_down = -cfg["confirm_alt_m"]

    logger.info(
        f"[Lawnmower] Descending to confirmed marker — "
        f"N={north:.2f} E={east:.2f} at {cfg['confirm_alt_m']}m"
    )

    # Descend to confirm altitude over the marker
    navigate_to(
        master, vo, cfg,
        north, east, confirm_down,
        label="marker-approach",
        marker_confirmed=marker_confirmed,
        stop_event=stop_event,
    )

    if stop_event.is_set():
        return

    # Log VO position at landing point for post-flight analysis
    final_north, final_east = vo_north_east(vo)
    logger.info(
        f"[Lawnmower] At confirm altitude — "
        f"VO position N={final_north:.2f} E={final_east:.2f}"
    )

    mission_state["valid_marker_confirmed"]    = True
    mission_state["confirmed_marker_position"] = (final_north, final_east)

    if on_confirmed:
        on_confirmed(final_north, final_east)

    # ------------------------------------------------------------------
    # LANDING — using StationaryLandingController
    #
    # TODO: When AprilTag precision landing is re-enabled in main.py,
    #       comment out the two lines below. The lawnmower will just
    #       hold position here and run_landing will take over via the
    #       marker_confirmed Event.
    # ------------------------------------------------------------------
    logger.info("[Lawnmower] Calling stationary_landing()")
    controller.stationary_landing()
    controller.disarm_motors()
    # ------------------------------------------------------------------

    stop_event.set()


def run_flight_loop(master, vo, cfg: dict,
                    waypoints: list,
                    controller: StationaryLandingController,
                    marker_confirmed,
                    stop_event: threading.Event,
                    on_confirmed,
                    mission_state: dict):
    """
    Main lawnmower flight loop — runs in a background thread.

    Iterates waypoints and polls marker_confirmed (multiprocessing.Event)
    set by run_vision in main.py. When set, snapshots the current VO
    position and calls fly_to_marker_and_land().
    """
    search_down = -cfg["search_alt_m"]
    total       = len(waypoints)

    for idx, (north, east) in enumerate(waypoints):
        if stop_event.is_set():
            break

        # Check before each waypoint in case vision fired between legs
        if marker_confirmed.is_set():
            logger.info(
                f"[Lawnmower] marker_confirmed before WP {idx + 1} "
                f"— aborting sweep"
            )
            break

        logger.info(
            f"[Lawnmower] WP {idx + 1}/{total} -> "
            f"N={north:.1f} E={east:.1f}"
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

    # Marker confirmed at some point during the sweep
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


def start_lawnmower_search(mav_master, vo,
                            valid_ids: list,
                            marker_confirmed,
                            controller: StationaryLandingController,
                            field_cfg: dict = None,
                            lm_cfg: dict = None,
                            on_confirmed=None) -> dict:
    """
    Launch the lawnmower flight in a background thread. Non-blocking.

    Parameters
    ----------
    mav_master       : pymavlink connection to Pixhawk
    vo               : VO adapter with .pose() -> (np.ndarray[x,y,z], yaw)
    valid_ids        : list of valid ArUco IDs (for logging)
    marker_confirmed : multiprocessing.Event set by run_vision in main.py
    controller       : StationaryLandingController for final land command
    field_cfg        : override FIELD_CONFIG
    lm_cfg           : override LAWNMOWER_CONFIG
    on_confirmed     : optional callback(north, east)

    Returns
    -------
    mission_state dict:
        {
            "valid_marker_confirmed":    bool,
            "confirmed_marker_position": (north, east) | None,
            "stop_event":                threading.Event,
        }
    """
    cfg = field_cfg or FIELD_CONFIG
    lm  = lm_cfg   or LAWNMOWER_CONFIG

    waypoints  = build_lawnmower_waypoints(cfg, lm)
    stop_event = threading.Event()

    mission_state = {
        "valid_marker_confirmed":    False,
        "confirmed_marker_position": None,
        "stop_event":                stop_event,
    }

    flight_thread = threading.Thread(
        target=run_flight_loop,
        args=(
            mav_master, vo, cfg,
            waypoints,
            controller,
            marker_confirmed,
            stop_event,
            on_confirmed,
            mission_state,
        ),
        daemon=True,
        name="LawnmowerFlight",
    )
    flight_thread.start()

    logger.info(
        f"[Lawnmower] Search started — "
        f"{len(waypoints)} waypoints, valid IDs={list(valid_ids)}"
    )
    return mission_state


def run_lawnmower_mission(mav_master, vo,
                           valid_ids: list,
                           marker_confirmed,
                           controller: StationaryLandingController,
                           field_cfg: dict = None,
                           lm_cfg: dict = None,
                           on_confirmed=None) -> dict:
    """
    Blocking version of start_lawnmower_search.

    Blocks until marker confirmed and landed, or field exhausted.
    Returns the mission_state dict.

    Usage:
        state = run_lawnmower_mission(
            mav_master=controller.master,
            vo=vo_adapter,
            valid_ids=[3, 7],
            marker_confirmed=marker_confirmed,
            controller=controller,
        )
        if state["valid_marker_confirmed"]:
            pos = state["confirmed_marker_position"]
    """
    mission_state = start_lawnmower_search(
        mav_master, vo,
        valid_ids=valid_ids,
        marker_confirmed=marker_confirmed,
        controller=controller,
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
            f"[LawnmowerRunner] Mission complete — "
            f"landed at N={pos[0]:.2f} E={pos[1]:.2f}"
        )
    else:
        logger.warning(
            "[LawnmowerRunner] Mission ended — valid marker not found"
        )

    return mission_state




# later on we can calculate fov based on oak intrinsics
# that means we can calculate column spacing based on altitude and desired overlap

'''
def compute_col_spacing(focal_length_px: float, frame_width_px: int,
                         search_alt_m: float,
                         overlap: float = 0.1) -> float:
    """
    Compute column spacing from camera intrinsics.

    overlap: fraction of footprint to overlap between columns (0.1 = 10%)
    A small overlap guarantees no gap even with slight VO drift.
    """
    fov_rad       = 2 * math.atan(frame_width_px / (2 * focal_length_px))
    footprint_w   = 2 * search_alt_m * math.tan(fov_rad / 2)
    col_spacing   = footprint_w * (1.0 - overlap)

    logger.info(
        f"[Lawnmower] FOV={math.degrees(fov_rad):.1f}° "
        f"footprint={footprint_w:.2f}m "
        f"col_spacing={col_spacing:.2f}m"
    )
    return col_spacing
    '''
