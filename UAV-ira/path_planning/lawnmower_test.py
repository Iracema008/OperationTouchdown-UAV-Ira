# landing/path_planning/lawnmower.py

''' Lawnmower path planning — standalone grid test

Run directly:
    python lawnmower.py
'''

import sys
import math
import time
import threading
from pymavlink import mavutil

from core.log import get_logger
from landing.pixhawk_controller.stationary_landing_controller import StationaryLandingController
from landing.geofence.geofence import upload_geofence_from_plan
from telemetry.telemetry_logger import log_event

logger = get_logger(__name__)


CONNECTION_STRING = "/dev/serial0"
BAUDRATE          = 57600
PLAN_PATH         = "/home/pi/mission/zone4.plan"
CSV_PATH          = "/home/pi/logs/lawnmower_test.csv"
FIELD_CONFIG = {
    "north_min_m":        0.0,
    "north_max_m":        8.0,
    "east_min_m":         0.0,
    "east_max_m":         8.0,
    "search_alt_m":       3.0,
    "confirm_alt_m":      1.2,
    "wp_accept_radius_m": 0.4,   # unused in fire-and-forget mode, kept for reference
    "wp_timeout_s":       20.0,  # fallback dwell for approach leg
    "move_speed_ms":      0.3,
    "approach_speed_ms":  0.2,   # slower speed used during marker descent
    "wp_dwell_s":         2.0,   # how long to broadcast each sweep waypoint
}

LAWNMOWER_CONFIG = {
    # Derived from cam intrinsics at 640x480, 3m altitude, 10 percent overlap
    "col_spacing_m": 2.8,
    "row_spacing_m": 2.5,
}

TAKEOFF_ALTITUDE  = FIELD_CONFIG["search_alt_m"]
DRONE_HEADING_DEG = 148  # change this later for whichever way its facing

# ALANS FUNCTION
def set_speed(master, speed_m_s: float, csv_path: str = None):
    logger.info(f"[Lawnmower] Setting speed limit to {speed_m_s} m/s")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
        0,
        1,           # Speed type: 1 = Ground Speed
        speed_m_s,
        -1,          # Throttle: -1 to ignore
        0, 0, 0, 0
    )
    if csv_path:
        log_event(csv_path, "SET_SPEED", {"speed_ms": speed_m_s})


def vo_north_east(vo) -> tuple:
    # VO pos layout: [x_right, y_down, z_forward]
    pos, _ = vo.pose()
    return float(pos[2]), float(pos[0])


def vo_full_pose(vo) -> dict:
    # Read full VIO pose as a loggable dict for CSV event rows
    # VO pos layout: [x_right, y_down, z_forward]
    pos, _ = vo.pose()
    return {
        "vio_north": round(float(pos[2]), 3),
        "vio_east": round(float(pos[0]), 3),
        "vio_down": round(float(pos[1]), 3),
    }


def send_goto_ned(master, north: float, east: float, down: float):
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
    # Generates grid waypoints in boustrophedon (snake) order
    # Returns list of (north, east) tuples
    waypoints = []
    col_spacing = lm_cfg["col_spacing_m"]
    row_spacing = lm_cfg["row_spacing_m"]

    north_min = cfg["north_min_m"]
    north_max = cfg["north_max_m"]
    east_min = cfg["east_min_m"]
    east_max = cfg["east_max_m"]

    east = east_min
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


def rotate_waypoints(waypoints: list, heading_deg: float) -> list:
    # Rotate grid to match drone's facing direction at EKF init.
    # heading_deg is the compass bearing the drone is facing (e.g. 148 for SE)
    angle   = math.radians(heading_deg)
    rotated = []
    for north, east in waypoints:
        new_north = north * math.cos(angle) - east * math.sin(angle)
        new_east  = north * math.sin(angle) + east * math.cos(angle)
        rotated.append((new_north, new_east))
    return rotated


def check_waypoints_in_bounds(waypoints: list, cfg: dict) -> bool:
    # After rotation the grid may expand outside the configured field boundary.
    # Check every waypoint and log any that fall out of bounds so we catch it
    # before arming rather than mid-flight.
    north_min = cfg["north_min_m"]
    north_max = cfg["north_max_m"]
    east_min = cfg["east_min_m"]
    east_max  = cfg["east_max_m"]

    all_ok = True

    for idx, (north, east) in enumerate(waypoints):
        out_north = north < north_min or north > north_max
        out_east  = east  < east_min  or east  > east_max

        if out_north or out_east:
            logger.warning(
                f"[Lawnmower] WP {idx + 1} out of bounds — "
                f"N={north:.2f} (limit {north_min}~{north_max}) "
                f"E={east:.2f} (limit {east_min}~{east_max})"
            )
            all_ok = False

    if all_ok:
        logger.info("[Lawnmower] All waypoints within bounds")
    else:
        logger.error(
            "[Lawnmower] Waypoints out of bounds — "
            "adjust DRONE_HEADING_DEG or field config before flying"
        )

    return all_ok


def navigate_to(master, vo, cfg: dict, north: float, east: float, down: float,
                label: str, stop_event: threading.Event,
                speed_m_s: float = None, dwell_s: float = None,
                csv_path: str = None) -> bool:
    # 1. Apply speed limit once at the start of each leg
    effective_speed = speed_m_s if speed_m_s is not None else cfg["move_speed_ms"]
    set_speed(master, effective_speed, csv_path=csv_path)

    # 2. How long to keep sending this setpoint before declaring done.
    #    dwell_s=None falls back to wp_timeout_s for the approach leg
    #    so that leg still waits long enough to physically arrive.
    hold_time = dwell_s if dwell_s is not None else cfg["wp_timeout_s"]
    t_start   = time.time()

    # 3. Snapshot VIO at the moment this setpoint is dispatched so the
    #    WP_START log row captures where the drone actually was when sent
    vio = vo_full_pose(vo)

    if csv_path:
        log_event(csv_path, "WP_START", {
            "label": label,
            "north": round(north, 2),
            "east":  round(east, 2),
            "down":  round(down, 2),
            "speed": effective_speed,
            "dwell": hold_time,
            **vio,
        })

    while not stop_event.is_set():
        send_goto_ned(master, north, east, down)

        # 4. Dwell complete — snapshot VIO again to capture where we ended up
        if time.time() - t_start >= hold_time:
            if csv_path:
                log_event(csv_path, "WP_DWELL_COMPLETE", {
                    "label": label,
                    **vo_full_pose(vo),
                })
            return True  # move on regardless of position

        time.sleep(0.1)

    # Only hits here if stop_event fired mid-dwell
    if csv_path:
        log_event(csv_path, "WP_ABORTED_STOP_EVENT", {
            "label": label,
            **vo_full_pose(vo),
        })
    return False


def run_flight_loop(master, vo, cfg: dict, waypoints: list,
                    controller: StationaryLandingController,
                    stop_event: threading.Event,
                    csv_path: str = None):
    search_down = -cfg["search_alt_m"]
    dwell_s     = cfg.get("wp_dwell_s", 2.0)
    total       = len(waypoints)

    log_event(csv_path, "FLIGHT_LOOP_START", {
        "total_wps":  total,
        "search_alt": cfg["search_alt_m"],
        "dwell_s":    dwell_s,
        "speed_ms":   cfg["move_speed_ms"],
    })

    for idx, (north, east) in enumerate(waypoints):
        if stop_event.is_set():
            break

        logger.info(f"[Lawnmower] WP {idx + 1}/{total} -> N={north:.1f} E={east:.1f}")

        navigate_to(
            master, vo, cfg,
            north, east, search_down,
            label=f"wp{idx + 1}",
            stop_event=stop_event,
            dwell_s=dwell_s,
            csv_path=csv_path,
        )

    # stop_event fired mid-sweep — skip landing, let main thread handle it
    if stop_event.is_set():
        logger.info("[Lawnmower] Flight loop stopped early by stop_event")
        log_event(csv_path, "FLIGHT_LOOP_STOPPED_EARLY", {**vo_full_pose(vo)})
        return

    # Normal sweep completion — land and disarm
    logger.info("[Lawnmower] Sweep complete — landing")
    log_event(csv_path, "SWEEP_COMPLETE", {**vo_full_pose(vo)})

    controller.stationary_landing()
    log_event(csv_path, "LANDED")

    controller.disarm_motors()
    log_event(csv_path, "DISARMED")

    stop_event.set()


if __name__ == "__main__":

    logger.info("[Lawnmower] === Standalone grid test starting ===")

    # 1. Connect to Pixhawk
    controller = StationaryLandingController(CONNECTION_STRING, BAUDRATE)
    master = controller.master

    # 2. Upload geofence before anything else — if this fails we do not fly.
    #    ArduPilot enforces the fence natively at 400Hz after this point,
    #    no position polling needed from Python side.
    try:
        upload_geofence_from_plan(master, PLAN_PATH)
    except Exception as e:
        logger.error(f"[Lawnmower] Geofence upload failed: {e}")
        logger.error("[Lawnmower] Aborting — fix geofence before flying")
        sys.exit(1)

    # 3. Build waypoints in boustrophedon order, then rotate to match the
    #    drone's actual compass heading at EKF init (DRONE_HEADING_DEG).
    #    NED origin is set by the FC at power-on so the grid must align to it.
    waypoints = build_lawnmower_waypoints(FIELD_CONFIG, LAWNMOWER_CONFIG)
    waypoints = rotate_waypoints(waypoints, heading_deg=DRONE_HEADING_DEG)

    # 4. Verify all rotated waypoints fall within the configured field boundary.
    #    If any are out of bounds we abort before arming — cheaper than a fly-away.
    if not check_waypoints_in_bounds(waypoints, FIELD_CONFIG):
        logger.error("[Lawnmower] Aborting — fix DRONE_HEADING_DEG or field config")
        sys.exit(1)

    # 5. Set cruise speed before takeoff
    set_speed(master, FIELD_CONFIG["move_speed_ms"], csv_path=CSV_PATH)

    log_event(CSV_PATH, "LAWNMOWER_START", {
        "total_wps":   len(waypoints),
        "speed_ms":    FIELD_CONFIG["move_speed_ms"],
        "search_alt":  FIELD_CONFIG["search_alt_m"],
        "heading_deg": DRONE_HEADING_DEG,
        "plan":        PLAN_PATH,
    })

    # 6. Arm and take off
    controller.change_flight_mode("GUIDED")
    controller.arm_motors()
    controller.takeoff_to_altitude(TAKEOFF_ALTITUDE)

    log_event(CSV_PATH, "TAKEOFF_COMPLETE", {"alt_m": TAKEOFF_ALTITUDE})

    # 7. Build VO — reads calibration from shared memory same as run_vio_process
    from multiprocessing import shared_memory
    from vio_slam.vio import VO_LK
    import numpy as np

    shm_calib  = shared_memory.SharedMemory(name="oak_calib")
    local_calib = np.ndarray((3, 3), dtype=np.float64, buffer=shm_calib.buf).copy()
    shm_calib.close()

    vo = VO_LK(K=local_calib)
    logger.info("[Lawnmower] VO_LK initialised")

    stop_event = threading.Event()

    # 8. Run flight loop in a thread so KeyboardInterrupt on the main thread
    #    can still trigger a clean land rather than hard killing the process
    flight_thread = threading.Thread(
        target=run_flight_loop,
        args=(master, vo, FIELD_CONFIG, waypoints, controller, stop_event, CSV_PATH),
        daemon=True,
        name="LawnmowerFlight",
    )
    flight_thread.start()

    # 9. Block main thread and hand control back cleanly on interrupt
    try:
        while flight_thread.is_alive():
            flight_thread.join(timeout=1.0)
    except KeyboardInterrupt:
        logger.warning("[Lawnmower] KeyboardInterrupt — signaling stop and landing")
        log_event(CSV_PATH, "INTERRUPTED_BY_USER", {})
        stop_event.set()
        flight_thread.join(timeout=5.0)
        controller.stationary_landing()
        controller.disarm_motors()
        log_event(CSV_PATH, "EMERGENCY_LAND_COMPLETE")

    logger.info("[Lawnmower] === Test complete ===")