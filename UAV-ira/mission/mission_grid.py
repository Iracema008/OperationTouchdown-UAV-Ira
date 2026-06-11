# mission/mission_grid.py

'''Mission Grid creates lawnmower waypoints, reads pose & frames, runs aruco detection, & lands '''

import math
import time
import numpy as np
import cv2
from pathlib import Path
from multiprocessing import shared_memory
from pymavlink import mavutil

from core.state import UAVStateAccessor, FlightMode
from core.log import get_logger
from landing.pixhawk_controller.stationary_landing_controller import StationaryLandingController
from landing.geofence.geofence import upload_geofence_from_plan
from vision.detectors.detector_manager import DetectorManager
from vision.video.camera_coordinate_transformer import CameraCoordinateTransformer
from telemetry.telemetry_logger import log_event

logger = get_logger(__name__)

PLAN_PATH = "home/orangepi/OperationTouchdown/UAV-ira/landing/geofence/csufField.plan"


FIELD_CONFIG = {
    "north_min_m": 0.0,
    "north_max_m": 5.0,
    "east_min_m": 0.0,
    "east_max_m": 5.0,
    "search_alt_m": 3.0,
    "confirm_alt_m": 0.3,
    "wp_accept_radius_m": 0.4, # meters from waypoint to consider it reached
    "wp_dwell_s": 3.0,  # seconds at each waypoint before continuing
    "approach_timeout_s": 15.0,
    "move_speed_ms": 0.3,
}

# Separate bounds for the post-rotation check
BOUNDS_CHECK = {
    "north_min_m": -5.0,
    "north_max_m":  5.0,
    "east_min_m":  -5.0,
    "east_max_m":   5.0,
}

LAWNMOWER_CONFIG = {
    "col_spacing_m": 2.8,
    "row_spacing_m": 2.5,
}

# Compass bearing the drone faces at power-on / EKF init.
# NED origin is set by the FC at power-on so the grid must align to it.
# Set to 0 and comment out rotate_waypoints() call below if facing north.
DRONE_HEADING_DEG = 90

CONFIRM_THRESHOLD   = 3
UNCERTAIN_THRESHOLD = 2   # kept for ArUco consec tracking, not for replan
LOOP_HZ     = 30
LOOP_PERIOD = 1.0 / LOOP_HZ


def set_speed(master, speed_m_s: float):
    # Cap ArduPilot ground speed via MAV_CMD_DO_CHANGE_SPEED
    logger.info(f"[GRID] Setting speed limit to {speed_m_s} m/s")
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

def dist2d(a: tuple, b: tuple) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def send_goto_ned(master, north: float, east: float, down: float):
    # non blocking means we have to resend the same setpoint until we reach it,
    # otherwise Pixhawk will time out and switch back to loiter
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
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        north, east, down,
        0, 0, 0, 0, 0, 0, 0, 0,
    )


def build_grid_waypoints() -> list:
    # Creates waypoints in boustrophedon (snake) order
    waypoints   = []
    col_spacing = LAWNMOWER_CONFIG["col_spacing_m"]
    row_spacing = LAWNMOWER_CONFIG["row_spacing_m"]
    north_min   = FIELD_CONFIG["north_min_m"]
    north_max   = FIELD_CONFIG["north_max_m"]
    east_min    = FIELD_CONFIG["east_min_m"]
    east_max    = FIELD_CONFIG["east_max_m"]

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
        f"[GRID] Generated {len(waypoints)} waypoints "
        f"({col_idx} columns, {FIELD_CONFIG['north_max_m']}x"
        f"{FIELD_CONFIG['east_max_m']}m field)"
    )
    return waypoints


def rotate_waypoints(waypoints: list, heading_deg: float) -> list:
    # Rotate grid to match drone's facing direction at EKF init.
    # heading_deg is the compass bearing the drone is facing (e.g. 150 for SE).
    # Comment out the call below and set DRONE_HEADING_DEG = 0 if facing north.
    angle   = math.radians(heading_deg)
    rotated = []
    for north, east in waypoints:
        new_north = north * math.cos(angle) - east * math.sin(angle)
        new_east  = north * math.sin(angle) + east * math.cos(angle)
        rotated.append((new_north, new_east))
    return rotated


def check_waypoints_in_bounds(waypoints: list, bounds:dict) -> bool:
    north_min = bounds["north_min_m"]
    north_max = bounds["north_max_m"]
    east_min  = bounds["east_min_m"]
    east_max  = bounds["east_max_m"]
    # After rotation the grid may expand outside the configured field boundary.
    # Logs any out-of-bounds waypoints before arming so we catch it on the ground.


    all_ok = True

    for idx, (north, east) in enumerate(waypoints):
        out_north = north < north_min or north > north_max
        out_east  = east  < east_min  or east  > east_max

        if out_north or out_east:
            logger.warning(
                f"[GRID] WP {idx + 1} out of bounds — "
                f"N={north:.2f} (limit {north_min}~{north_max}) "
                f"E={east:.2f} (limit {east_min}~{east_max})"
            )
            all_ok = False

    if all_ok:
        logger.info("[GRID] All waypoints within bounds")
    else:
        logger.error(
            "[GRID] Waypoints out of bounds — "
            "adjust DRONE_HEADING_DEG or field config before flying"
        )

    return all_ok


def save_snapshot(frame, corners, ids, marker_id, log_timestamp, suffix=""):
    save_dir = Path("flight_logs/markers")
    save_dir.mkdir(parents=True, exist_ok=True)

    annotated = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    tag = f"_{suffix}" if suffix else ""

    filename = save_dir / f"marker_{marker_id}_{log_timestamp}{tag}.png"
    cv2.imwrite(str(filename), annotated)
    logger.info(f"[GRID] Snapshot → {filename}")


def run_mission(lock, marker_confirmed, ugv_signal, hover_reached, cfg, log_timestamp, uncertain_pos, planner, rgb_frame_mutex):
    # Arms, takes off, searches field, runs ArUco detection, confirms & lands
    logger.info("[GRID] Process starting")
    W, H  = cfg.camera.width, cfg.camera.height
    state = UAVStateAccessor(lock, marker_confirmed, ugv_signal, hover_reached)

    controller = StationaryLandingController(
        cfg.pixhawk.connection_string,
        cfg.pixhawk.baud_rate
    )

    # 1. Upload geofence before arming — if this fails we do not fly.
    #    ArduPilot enforces the fence natively at 400Hz after this point,
    #    no position polling needed from Python side.
    try:
        upload_geofence_from_plan(controller.master, PLAN_PATH)
    except Exception as e:
        logger.error(f"[GRID] Geofence upload failed: {e}")
        logger.error("[GRID] Aborting — fix geofence before flying")
        state.close()
        return

    # 2. Build waypoints, rotate to match drone heading at EKF init, then
    #    verify all rotated points fall within the field boundary before arming.
    #    Comment out rotate_waypoints() and set DRONE_HEADING_DEG = 0 if facing north.
    waypoints = build_grid_waypoints()
    waypoints = rotate_waypoints(waypoints, heading_deg=DRONE_HEADING_DEG)

    if not check_waypoints_in_bounds(waypoints, BOUNDS_CHECK):
        logger.error("[GRID] Aborting — fix DRONE_HEADING_DEG or field config")
        state.close()
        return

    # 3. Arm and takeoff
    logger.info("[GRID] Arming and taking off")
    controller.change_flight_mode("GUIDED")
    set_speed(controller.master, FIELD_CONFIG["move_speed_ms"])
    controller.arm_motors()
    controller.takeoff_to_altitude(cfg.pixhawk.hover_altitude_m)
    state.set_flight_mode(FlightMode.SCAN)

    takeoff_start = time.time()
    logger.info("[GRID] Airborne — starting mission loop")

    # 4. Check if we are in simulated mode, check for connection string udp in core/config
    sitl_mode  = not cfg.pixhawk.connection_string.startswith("/dev")
    shm_rgb    = None
    shared_rgb = None
    local_rgb  = np.zeros((H, W, 3), dtype=np.uint8)

    if not sitl_mode:
        try:
            shm_rgb    = shared_memory.SharedMemory(name="oak_rgb")
            shared_rgb = np.ndarray((H, W, 3), dtype=np.uint8, buffer=shm_rgb.buf)
        except Exception as e:
            logger.warning(f"[GRID] oak_rgb unavailable: {e} — ArUco disabled")
            sitl_mode = True

    if sitl_mode:
        logger.info("[GRID] SITL mode — position from LOCAL_POSITION_NED")
        controller.master.mav.request_data_stream_send(
            controller.master.target_system,
            controller.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,
            10, 1
        )

    # 5. ArUco setup
    target_ids = cfg.aruco.target_marker_id
    if not isinstance(target_ids, list):
        target_ids = [target_ids]

    detector      = DetectorManager(cfg.detector).get_detector()
    transformer   = CameraCoordinateTransformer(cfg.video)
    consec_counts = {}

    # 6. Waypoint state
    total       = len(waypoints)
    wp_idx      = 0
    wp_start_t  = time.time()
    search_down = -FIELD_CONFIG["search_alt_m"]

    # Phase tracker — sweep → approach → land → done
    phase          = "sweep"
    landing_north  = 0.0
    landing_east   = 0.0
    approach_start = 0.0

    # Print full planned path at startup
    logger.info("[GRID] === PLANNED PATH ===")
    for i, (n, e) in enumerate(waypoints):
        logger.info(f"[GRID]   WP {i + 1:>2}/{total} → N={n:.1f} E={e:.1f}")
    logger.info("[GRID] === END PATH ===")
    logger.info(
        f"[GRID] Search starting — {total} waypoints, "
        f"dwell={FIELD_CONFIG['wp_dwell_s']}s each, "
        f"target IDs={target_ids}, "
        f"heading={DRONE_HEADING_DEG}deg"
    )

    try:
        while phase != "done":
            t_start = time.time()

            # 7. Position
            if sitl_mode:
                pos_msg = controller.master.recv_match(
                    type='LOCAL_POSITION_NED', blocking=False
                )
                if pos_msg:
                    cur_north = float(pos_msg.x)
                    cur_east  = float(pos_msg.y)
                    state.set_vio_position(
                        cur_north, cur_east, float(pos_msg.z), 0.0
                    )
                else:
                    (vio_x, vio_y, _, _), _ = state.get_vio_position()
                    cur_north = vio_x
                    cur_east  = vio_y
            else:
                (vio_x, vio_y, _, _), _ = state.get_vio_position()
                cur_north = vio_x
                cur_east  = vio_y

            # 8. RGB frame
            if not sitl_mode and shared_rgb is not None:
                with rgb_frame_mutex:
                    np.copyto(local_rgb, shared_rgb)

            # 9. ArUco detection — runs every frame until marker confirmed
            if not sitl_mode and not marker_confirmed.is_set():
                corners, ids, _ = detector.detect(local_rgb)

                if ids is None:
                    consec_counts.clear()
                else:
                    flat_ids = ids.flatten().tolist()
                    matched  = [i for i in flat_ids if i in target_ids]

                    for mid in list(consec_counts):
                        if mid not in flat_ids:
                            consec_counts[mid] = 0

                    for mid in matched:
                        consec_counts[mid] = consec_counts.get(mid, 0) + 1
                        count = consec_counts[mid]

                        logger.debug(
                            f"[GRID] ID={mid} consec={count}/{CONFIRM_THRESHOLD}"
                        )

                        if count >= CONFIRM_THRESHOLD:
                            total_flight_time = time.time() - takeoff_start
                            if total_flight_time < 5.0:
                                logger.info(
                                    f"[GRID] Marker confirmed but min flight time not met "
                                    f"({total_flight_time:.1f}s / 5.0s) — continuing search"
                                )
                                continue
                            cx = (corners[0][0][0][0] +
                                  (corners[0][0][2][0] - corners[0][0][0][0]) / 2)
                            cy = (corners[0][0][0][1] +
                                  (corners[0][0][2][1] - corners[0][0][0][1]) / 2)
                            x, y, z = transformer.transform(
                                (cx, cy), FIELD_CONFIG["search_alt_m"]
                            )
                            save_snapshot(
                                local_rgb, corners, ids, mid, log_timestamp
                            )
                            state.set_aruco_pose(x, y, z, mid)
                            logger.info(
                                f"[GRID] ArUco ID={mid} CONFIRMED — "
                                f"VO N={cur_north:.2f} E={cur_east:.2f}"
                            )
                            landing_north  = cur_north
                            landing_east   = cur_east
                            approach_start = time.time()
                            phase = "approach"

            # 10. Phase execution
            if phase == "sweep":
                if wp_idx < total:
                    tgt_n, tgt_e = waypoints[wp_idx]

                    send_goto_ned(controller.master, tgt_n, tgt_e, search_down)

                    elapsed = time.time() - wp_start_t

                    if elapsed >= FIELD_CONFIG["wp_dwell_s"]:
                        dist = dist2d((cur_north, cur_east), (tgt_n, tgt_e))

                        if wp_idx + 1 < total:
                            next_n, next_e = waypoints[wp_idx + 1]
                            logger.info(
                                f"[GRID] Departing WP {wp_idx + 1}/{total} | "
                                f"from N={cur_north:.2f} E={cur_east:.2f} | "
                                f"to N={next_n:.1f} E={next_e:.1f} | "
                                f"dist to next={dist2d((cur_north, cur_east), (next_n, next_e)):.2f}m"
                            )
                        else:
                            logger.info(
                                f"[GRID] Departing WP {wp_idx + 1}/{total} — final waypoint | "
                                f"from N={cur_north:.2f} E={cur_east:.2f} | "
                                f"dist from target={dist:.2f}m"
                            )

                        wp_idx    += 1
                        wp_start_t = time.time()
                else:
                    logger.info("[GRID] Sweep complete — marker not found")
                    phase = "done"

            elif phase == "approach":
                confirm_down = -FIELD_CONFIG["confirm_alt_m"]
                send_goto_ned(
                    controller.master,
                    landing_north, landing_east, confirm_down
                )
                dist = dist2d(
                    (cur_north, cur_east),
                    (landing_north, landing_east)
                )
                if dist <= FIELD_CONFIG["wp_accept_radius_m"]:
                    logger.info(
                        f"[GRID] Over marker at {FIELD_CONFIG['confirm_alt_m']}m"
                    )
                    phase = "land"
                elif time.time() - approach_start > FIELD_CONFIG["approach_timeout_s"]:
                    logger.warning("[GRID] Approach timeout — landing anyway")
                    phase = "land"

            elif phase == "land":
                logger.info("[GRID] Stationary landing")
                # Uncomment for AprilTag landing:
                # _run_apriltag_landing(controller, shared_rgb, local_rgb, rgb_frame_mutex, cfg)
                controller.stationary_landing()
                controller.disarm_motors()
                state.set_flight_mode(FlightMode.LAND)
                phase = "done"

            # 11. Loop timing
            elapsed_s = time.time() - t_start
            time.sleep(max(0.0, LOOP_PERIOD - elapsed_s))

    except KeyboardInterrupt:
        logger.warning("[GRID] Interrupted")

    finally:
        if shm_rgb is not None:
            shm_rgb.close()
        state.close()
        logger.info("[GRID] Process exiting")