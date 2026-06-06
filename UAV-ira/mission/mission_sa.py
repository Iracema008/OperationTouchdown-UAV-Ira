''' Simulated Annealing lawnmower mission, no arrival checking.

SA runs ONCE pre-flight to reorder the grid waypoints into the
shortest total path. After takeoff the drone follows that fixed
optimized order. No mid-flight replan.

Loop structure (30Hz non-blocking):
    1. Read position  — LOCAL_POSITION_NED (SITL) or uav_vio (real flight)
    2. Read RGB frame — shared memory written by broadcaster
    3. ArUco detection — every frame, 3 consecutive → confirmed → approach
    4. Phase execution — sweep / approach / land / done
    5. Sleep to hit 33ms budget

SITL testing:
    Press Enter in the run_sitl.py terminal to simulate marker_confirmed.
    The shared marker_confirmed Event crosses process boundaries correctly.
    Drone stops sweeping and transitions to approach → land at current position.

UART0: sole owner for arm → sweep → land.
'''

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
from path_planning.simulated_annealing import build_sa_waypoints
from vision.detectors.detector_manager import DetectorManager
from vision.video.camera_coordinate_transformer import CameraCoordinateTransformer
from telemetry.telemetry_logger import log_event

logger = get_logger(__name__)

FIELD_N = 8.0   # north dimension — change both to match actual field
FIELD_E = 8.0   # east dimension

FIELD_CONFIG = {
    "north_min_m":        0.0,
    "north_max_m":        FIELD_N,
    "east_min_m":         0.0,
    "east_max_m":         FIELD_E,
    "search_alt_m":       3.0,
    "confirm_alt_m":      1.2,
    "wp_accept_radius_m": 0.4,
    "wp_dwell_s":         3.0,
    "approach_timeout_s": 15.0,
}

LAWNMOWER_CONFIG = {
    "col_spacing_m": 2.8,
    "row_spacing_m": 2.5,
}

CONFIRM_THRESHOLD = 3
MIN_FLIGHT_TIME_S = 5.0
LOOP_HZ           = 30
LOOP_PERIOD       = 1.0 / LOOP_HZ


def dist2d(a: tuple, b: tuple) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def send_goto_ned(master, north: float, east: float, down: float):
    '''Send NED position setpoint — returns immediately (non-blocking).
    Non-blocking means we resend every loop tick until the dwell timer fires,
    otherwise Pixhawk times out and switches back to loiter.
    '''
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
    return waypoints


def save_snapshot(frame, corners, ids, marker_id, log_timestamp, suffix=""):
    '''Save annotated ArUco detection frame to flight_logs/markers/.'''
    save_dir = Path("flight_logs/markers")
    save_dir.mkdir(parents=True, exist_ok=True)
    annotated = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    tag       = f"_{suffix}" if suffix else ""
    filename  = save_dir / f"marker_{marker_id}_{log_timestamp}{tag}.png"
    cv2.imwrite(str(filename), annotated)
    logger.info(f"[SA] Snapshot → {filename}")


def run_mission(lock, marker_confirmed, ugv_signal, hover_reached, cfg,
                log_timestamp, uncertain_pos, planner, rgb_frame_mutex):
    '''SA lawnmower mission process.

    SA runs once pre-flight to optimize waypoint order.
    Arms, takes off, searches field in SA-optimized order.
    Advances waypoints by dwell time — no arrival check.
    ArUco detection every frame → confirmed → approach → land.
    In SITL press Enter in run_sitl.py to simulate marker found.
    '''
    logger.info("[SA] Process starting")

    W, H     = cfg.camera.width, cfg.camera.height
    state    = UAVStateAccessor(lock, marker_confirmed, ugv_signal, hover_reached)
    csv_path = f"flight_logs/flight_{log_timestamp}.csv"

    controller = StationaryLandingController(
        cfg.pixhawk.connection_string,
        cfg.pixhawk.baud_rate
    )

    # 1. SA pre-flight optimization — reorders grid waypoints to minimize
    #    total path distance. Runs before arming so no flight time is lost.
    logger.info("[SA] Running pre-flight path optimization...")
    grid      = build_grid_waypoints()
    waypoints = build_sa_waypoints(grid, start_pos=(0.0, 0.0))
    total     = len(waypoints)

    logger.info("[SA] === OPTIMIZED PATH ===")
    for i, (n, e) in enumerate(waypoints):
        logger.info(f"[SA]   WP {i + 1:>2}/{total} → N={n:.1f} E={e:.1f}")
    logger.info("[SA] === END PATH ===")

    # 2. Arm and takeoff
    logger.info("[SA] Arming and taking off")
    controller.change_flight_mode("GUIDED")
    controller.arm_motors()
    controller.takeoff_to_altitude(cfg.pixhawk.hover_altitude_m)
    state.set_flight_mode(FlightMode.SCAN)
    takeoff_start = time.time()
    log_event(csv_path, "uav_start")   # Appendix B — UAV Start Time
    logger.info("[SA] Airborne — starting mission loop")

    # 3. SITL detection — no OAK-D on Mac so camera is unavailable.
    #    In SITL, marker_confirmed is set by run_sitl.py when Enter is pressed.
    #    This is a multiprocessing.Event so it crosses process boundaries correctly.
    #    A threading.Event inside a spawned process does NOT work — stdin is closed
    #    in spawned subprocesses and input() throws EOFError immediately.
    sitl_mode  = not cfg.pixhawk.connection_string.startswith("/dev")
    shm_rgb    = None
    shared_rgb = None
    local_rgb  = np.zeros((H, W, 3), dtype=np.uint8)

    if not sitl_mode:
        # Connect to RGB shared memory written by broadcaster
        try:
            shm_rgb    = shared_memory.SharedMemory(name="oak_rgb")
            shared_rgb = np.ndarray((H, W, 3), dtype=np.uint8, buffer=shm_rgb.buf)
        except Exception as e:
            logger.warning(f"[SA] oak_rgb unavailable: {e} — ArUco disabled")
            sitl_mode = True

    if sitl_mode:
        logger.info("[SA] SITL mode — position from LOCAL_POSITION_NED")
        logger.info("[SA] SITL mode — press Enter in run_sitl.py to simulate marker found")
        controller.master.mav.request_data_stream_send(
            controller.master.target_system,
            controller.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,
            10, 1
        )

    # ArUco
    target_ids = cfg.aruco.target_marker_id
    if not isinstance(target_ids, list):
        target_ids = [target_ids]

    detector      = DetectorManager(cfg.detector).get_detector()
    transformer   = CameraCoordinateTransformer(cfg.video)
    consec_counts = {}

    wp_idx         = 0
    wp_start_t     = time.time()
    search_down    = -FIELD_CONFIG["search_alt_m"]
    phase          = "sweep"
    landing_north  = 0.0
    landing_east   = 0.0
    approach_start = 0.0

    logger.info(
        f"[SA] Search starting — {total} waypoints, "
        f"dwell={FIELD_CONFIG['wp_dwell_s']}s each, "
        f"target IDs={target_ids}"
    )

    try:
        while phase != "done":
            t_start = time.time()

            # 4. Position
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

            # 5. RGB frame
            if not sitl_mode and shared_rgb is not None:
                with rgb_frame_mutex:
                    np.copyto(local_rgb, shared_rgb)

            # 6. SITL marker trigger — checks marker_confirmed set by run_sitl.py.
            #    Uses the shared multiprocessing.Event which works across processes.
            #    Sets landing position to current VIO position then transitions to
            #    approach, identical to what real ArUco confirmation does.
            if sitl_mode and marker_confirmed.is_set() and phase == "sweep":
                airborne_time = time.time() - takeoff_start
                if airborne_time < MIN_FLIGHT_TIME_S:
                    logger.info(
                        f"[SA] Min flight time not met "
                        f"({airborne_time:.1f}s / {MIN_FLIGHT_TIME_S}s) — holding"
                    )
                else:
                    logger.info(
                        f"[SA] SITL marker confirmed at "
                        f"N={cur_north:.2f} E={cur_east:.2f}"
                    )
                    landing_north  = cur_north
                    landing_east   = cur_east
                    approach_start = time.time()
                    log_event(csv_path, "destination_discovered", {
                        "marker_id": "SITL",
                        "north":     round(cur_north, 2),
                        "east":      round(cur_east, 2),
                    })
                    phase = "approach"

            # 7. ArUco detection — real hardware only, skipped in SITL
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
                            f"[SA] ID={mid} consec={count}/{CONFIRM_THRESHOLD}"
                        )

                        if count >= CONFIRM_THRESHOLD:
                            # Enforce minimum 5 second flight time per competition rules
                            airborne_time = time.time() - takeoff_start
                            if airborne_time < MIN_FLIGHT_TIME_S:
                                logger.info(
                                    f"[SA] Min flight time not met "
                                    f"({airborne_time:.1f}s / {MIN_FLIGHT_TIME_S}s) — continuing"
                                )
                                continue

                            cx = (corners[0][0][0][0] +
                                  (corners[0][0][2][0] - corners[0][0][0][0]) / 2)
                            cy = (corners[0][0][0][1] +
                                  (corners[0][0][2][1] - corners[0][0][0][1]) / 2)
                            x, y, z = transformer.transform(
                                (cx, cy), FIELD_CONFIG["search_alt_m"]
                            )
                            save_snapshot(local_rgb, corners, ids, mid, log_timestamp)
                            state.set_aruco_pose(x, y, z, mid)
                            logger.info(
                                f"[SA] ArUco ID={mid} CONFIRMED — "
                                f"VO N={cur_north:.2f} E={cur_east:.2f}"
                            )
                            landing_north  = cur_north
                            landing_east   = cur_east
                            approach_start = time.time()
                            log_event(csv_path, "destination_discovered", {
                                "marker_id": mid,
                                "north":     round(cur_north, 2),
                                "east":      round(cur_east, 2),
                            })
                            phase = "approach"

            # 8. Phase execution
            if phase == "sweep":
                if wp_idx < total:
                    tgt_n, tgt_e = waypoints[wp_idx]
                    send_goto_ned(controller.master, tgt_n, tgt_e, search_down)
                    elapsed = time.time() - wp_start_t

                    if elapsed >= FIELD_CONFIG["wp_dwell_s"]:
                        dist_from_target = dist2d(
                            (cur_north, cur_east), (tgt_n, tgt_e)
                        )
                        if wp_idx + 1 < total:
                            next_n, next_e = waypoints[wp_idx + 1]
                            logger.info(
                                f"[SA] Departing WP {wp_idx + 1}/{total} | "
                                f"from N={cur_north:.2f} E={cur_east:.2f} | "
                                f"to N={next_n:.1f} E={next_e:.1f} | "
                                f"dist to next={dist2d((cur_north, cur_east), (next_n, next_e)):.2f}m"
                            )
                        else:
                            logger.info(
                                f"[SA] Departing WP {wp_idx + 1}/{total} — final | "
                                f"from N={cur_north:.2f} E={cur_east:.2f} | "
                                f"dist from target={dist_from_target:.2f}m"
                            )
                        wp_idx    += 1
                        wp_start_t = time.time()
                else:
                    logger.info("[SA] Search complete — marker not found")
                    phase = "done"

            # Fly to confirmed marker position at lower altitude before landing
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
                        f"[SA] Over marker at {FIELD_CONFIG['confirm_alt_m']}m"
                    )
                    phase = "land"
                elif time.time() - approach_start > FIELD_CONFIG["approach_timeout_s"]:
                    logger.warning("[SA] Approach timeout — landing anyway")
                    phase = "land"

            # Landing — uncomment AprilTag precision landing when ready
            elif phase == "land":
                logger.info("[SA] Stationary landing")
                # _run_apriltag_landing(controller, shared_rgb, local_rgb, rgb_frame_mutex, cfg)
                controller.stationary_landing()
                controller.disarm_motors()
                state.set_flight_mode(FlightMode.LAND)
                phase = "done"

            # 9. Loop timing
            elapsed_s = time.time() - t_start
            time.sleep(max(0.0, LOOP_PERIOD - elapsed_s))

    except KeyboardInterrupt:
        logger.warning("[SA] Interrupted")

    finally:
        log_event(csv_path, "uav_end")   # Appendix B — UAV End Time
        if shm_rgb is not None:
            shm_rgb.close()
        state.close()
        logger.info("[SA] Process exiting")