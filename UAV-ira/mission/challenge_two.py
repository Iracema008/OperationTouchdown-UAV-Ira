''' Challenge 2: UAV autonomously launched and identifies the ArUco marker’s location
UGV starts traveling to the destination and the UAV lands on moving UGV system
Entire System arrives and stops at the destination. '''

# python main.py --mode scan --planner c2

import time
import math
import numpy as np
import cv2
from multiprocessing import shared_memory
from pymavlink import mavutil
from pupil_apriltags import Detector as AprilDetector

from core.state import UAVStateAccessor, FlightMode
from core.log import get_logger
from landing.pixhawk_controller.stationary_landing_controller import StationaryLandingController
from controls.lora_sender import send_goto, send_drive_c1, send_stop
from vision.detectors.detector_manager import DetectorManager
from vision.video.camera_coordinate_transformer import CameraCoordinateTransformer

logger = get_logger(__name__)


TAKEOFF_ALTITUDE_M = 2.0    # above 4 foot minimum (1.22m)
MIN_FLIGHT_TIME_S  = 5.0    # challenge requires minimum 5 seconds airborne
MISSION_TIMEOUT_S  = 600.0  # 10 minutes from first UGV movement to landing
WAIT_ON_UGV_S      = 10.0   # sit on UGV after landing without separating

FIELD_CONFIG = {
    "north_min_m":        0.0,
    "north_max_m":        8.0,
    "east_min_m":         0.0,
    "east_max_m":         8.0,
    "search_alt_m":       3.0,
    "confirm_alt_m":      1.2,
    "wp_accept_radius_m": 0.4,
    "wp_dwell_s":         3.0,
    "approach_timeout_s": 15.0,
}
LAWNMOWER_CONFIG = {
    "col_spacing_m": 2.8,
    "row_spacing_m": 2.0,
}
ARUCO_CONFIRM_THRESHOLD = 3  # consecutive detections before communicating to UGV

# Return to home
HOME_ACCEPT_RADIUS_M = 1.0   # loose radius — UGV platform has physical size
HOME_TIMEOUT_S       = 15.0  # if home not reached in time, proceed to forward anyway

# Forward search phase
FORWARD_SPEED_MS = 0.5  # m/s in body frame,heading at takeoff does not matter

# AprilTag detection
# switched to a smaller 5.5 by 5.5 aruco marker
TARGET_TAG_ID = 67
TAG_SIZE_M = 0.14

# Tag tracking and landing
LANDING_THRESHOLD_M = 0.4
HOVER_TIMEOUT_S     = 3.0
SEARCH_TIMEOUT_S    = 7.0
COAST_BOOST         = 1.3

LOOP_HZ = 30
LOOP_PERIOD = 1.0 / LOOP_HZ


def dist2d(a: tuple, b: tuple) -> float:
    '''Euclidean distance between two (north, east) points.'''
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def send_goto_ned(master, north: float, east: float, down: float):
    '''Send NED position setpoint — returns immediately (non-blocking).'''
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
    '''Boustrophedon column sweep — returns (north, east) tuples in sweep order.'''
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

    logger.info(f"[C2] Generated {len(waypoints)} waypoints ({col_idx} columns)")
    return waypoints


def run_mission(lock, marker_confirmed, ugv_signal, hover_reached,
                cfg, log_timestamp, uncertain_pos, planner, rgb_frame_mutex):
    '''Challenge 2 mission process.

    Phase sequence:
        sweep       → grid lawnmower to find ArUco marker
        communicate → confirm marker, send NED location to UGV
        return      → fly back to start position above UGV
        forward     → fly forward in body frame until AprilTag on UGV detected
        track       → chase UGV with PID velocity control
        land        → smart_touchdown once centered above AprilTag
        wait        → disarmed on moving UGV for 10 seconds
        done        → mission complete
    '''
    logger.info("[C2] Process starting")

    W, H  = cfg.camera.width, cfg.camera.height
    state = UAVStateAccessor(lock, marker_confirmed, ugv_signal, hover_reached)

    controller = StationaryLandingController(
        cfg.pixhawk.connection_string,
        cfg.pixhawk.baud_rate
    )

    # 1. Arm and take off. UGV is stationary at this point.
    logger.info("[C2] Arming and taking off")
    controller.change_flight_mode("GUIDED")
    controller.arm_motors()
    controller.takeoff_to_altitude(TAKEOFF_ALTITUDE_M)
    state.set_flight_mode(FlightMode.SCAN)
    takeoff_time  = time.time()
    mission_start = 0.0   # starts when UGV begins moving (after communicate phase)
    logger.info(f"[C2] Airborne at {TAKEOFF_ALTITUDE_M}m — minimum flight time {MIN_FLIGHT_TIME_S}s starts now")

    # 2. Connect to RGB shared memory for ArUco detection during sweep,
    #    and camera calibration for AprilTag detection during landing.
    #    Broadcaster owns the OAK-D — we only read from shared memory here.
    sitl_mode  = not cfg.pixhawk.connection_string.startswith("/dev")
    shm_rgb    = None
    shm_calib  = None
    shared_rgb = None
    local_rgb  = np.zeros((H, W, 3), dtype=np.uint8)
    april      = None
    FX = FY = CX = CY = 0.0

    target_ids = cfg.aruco.target_marker_id
    if not isinstance(target_ids, list):
        target_ids = [target_ids]

    detector    = DetectorManager(cfg.detector).get_detector()
    transformer = CameraCoordinateTransformer(cfg.video)

    if not sitl_mode:
        try:
            shm_rgb    = shared_memory.SharedMemory(name="oak_rgb")
            shared_rgb = np.ndarray((H, W, 3), dtype=np.uint8, buffer=shm_rgb.buf)
            shm_calib  = shared_memory.SharedMemory(name="oak_calib")
            calib_mat  = np.ndarray((3, 3), dtype=np.float64, buffer=shm_calib.buf)
            FX = calib_mat[0, 0]
            FY = calib_mat[1, 1]
            CX = calib_mat[0, 2]
            CY = calib_mat[1, 2]
            april = AprilDetector(
                families="tag36h11",
                nthreads=1,
                quad_decimate=2.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25
            )
            logger.info(f"[C2] Detectors ready — fx={FX:.2f} fy={FY:.2f} cx={CX:.2f} cy={CY:.2f}")
        except Exception as e:
            logger.warning(f"[C2] Camera shared memory unavailable: {e} — treating as SITL")
            sitl_mode = True

    if sitl_mode:
        logger.info("[C2] SITL mode — camera disabled")

    # 3. Build waypoints and initialise phase state machine
    waypoints      = build_grid_waypoints()
    total          = len(waypoints)
    wp_idx         = 0
    wp_start_t     = time.time()
    search_down    = -FIELD_CONFIG["search_alt_m"]
    consec_counts  = {}

    phase              = "sweep"
    marker_north       = 0.0   # NED position of confirmed ArUco marker
    marker_east        = 0.0
    home_start_t       = 0.0
    last_tag_time      = time.time()
    is_escaping_ground = False
    escape_target_z    = 0.0
    wait_start         = 0.0

    logger.info(f"[C2] Sweep starting — {total} waypoints, target IDs={target_ids}")

    try:
        while phase != "done":
            t_start = time.time()

            # 10 minute mission timeout starts when UGV begins moving (after communicate)
            if mission_start > 0 and (time.time() - mission_start > MISSION_TIMEOUT_S):
                logger.error("[C2] Mission timeout (10 min) — emergency landing")
                controller.smart_touchdown(timeout=8.0)
                controller.disarm_motors()
                break

            # 4. Read RGB frame from shared memory every tick.
            #    ArUco detection runs during sweep phase.
            #    AprilTag detection runs during forward and track phases.
            if not sitl_mode and shared_rgb is not None:
                with rgb_frame_mutex:
                    np.copyto(local_rgb, shared_rgb)

            # 5. SWEEP phase — grid lawnmower over the field looking for ArUco marker.
            #    Advances waypoints on dwell timer, no arrival check.
            #    3 consecutive detections confirms the marker and triggers communicate.
            if phase == "sweep":
                aruco_pose = None
                if not sitl_mode:
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
                            logger.debug(f"[C2] ArUco ID={mid} consec={count}/{ARUCO_CONFIRM_THRESHOLD}")
                            if count >= ARUCO_CONFIRM_THRESHOLD:
                                # Read current VIO position as marker location
                                (vio_x, vio_y, _, _), _ = state.get_vio_position()
                                marker_north = vio_x
                                marker_east  = vio_y
                                logger.info(
                                    f"[C2] ArUco ID={mid} CONFIRMED — "
                                    f"marker at N={marker_north:.2f} E={marker_east:.2f}"
                                )
                                phase = "communicate"

                if phase == "sweep":
                    if wp_idx < total:
                        tgt_n, tgt_e = waypoints[wp_idx]
                        send_goto_ned(controller.master, tgt_n, tgt_e, search_down)
                        elapsed = time.time() - wp_start_t
                        if elapsed >= FIELD_CONFIG["wp_dwell_s"]:
                            (cur_north, cur_east, _, _), _ = state.get_vio_position()
                            if wp_idx + 1 < total:
                                next_n, next_e = waypoints[wp_idx + 1]
                                logger.info(
                                    f"[C2] Departing WP {wp_idx + 1}/{total} | "
                                    f"from N={cur_north:.2f} E={cur_east:.2f} | "
                                    f"to N={next_n:.1f} E={next_e:.1f} | "
                                    f"dist to next={dist2d((cur_north, cur_east), (next_n, next_e)):.2f}m"
                                )
                            else:
                                logger.info(f"[C2] Departing WP {wp_idx + 1}/{total} — final waypoint")
                            wp_idx    += 1
                            wp_start_t = time.time()
                    else:
                        # Sweep complete without finding marker — emergency land
                        logger.error("[C2] Sweep complete — marker not found, emergency landing")
                        controller.stationary_landing()
                        controller.disarm_motors()
                        phase = "done"

            # 6. COMMUNICATE phase — send confirmed marker NED coordinates to UGV.
            #    UGV will begin moving toward the marker after receiving this.
            #    Mission timeout clock starts here.
            elif phase == "communicate":
                send_goto(marker_north, marker_east, cfg)
                mission_start = time.time()   # 10 min clock starts now
                logger.info("[C2] UGV signalled — returning to start position")
                home_start_t = time.time()
                phase = "return"

            # 7. RETURN phase — fly back to (0, 0) where the UGV is parked.
            #    Uses NED position setpoints. Loose 1.0m arrival radius since the
            #    UGV platform has physical size and VIO may have drifted slightly.
            #    15 second timeout — if not home by then, proceed to forward anyway
            #    and try to find the AprilTag.
            elif phase == "return":
                send_goto_ned(controller.master, 0.0, 0.0, -TAKEOFF_ALTITUDE_M)
                (cur_north, cur_east, _, _), _ = state.get_vio_position()
                dist_home = dist2d((cur_north, cur_east), (0.0, 0.0))
                elapsed   = time.time() - home_start_t

                if dist_home <= HOME_ACCEPT_RADIUS_M:
                    logger.info(f"[C2] Home reached (dist={dist_home:.2f}m) — switching to FORWARD")
                    phase = "forward"
                elif elapsed > HOME_TIMEOUT_S:
                    logger.warning(
                        f"[C2] Home timeout ({HOME_TIMEOUT_S}s) — "
                        f"dist={dist_home:.2f}m — proceeding to FORWARD anyway"
                    )
                    phase = "forward"
                else:
                    logger.debug(f"[C2] Returning home — dist={dist_home:.2f}m elapsed={elapsed:.1f}s")

            # 8. FORWARD phase — fly forward in body frame until AprilTag on UGV
            #    enters the frame. UGV is now moving toward the marker so it will
            #    come into view. Body frame means heading at takeoff does not matter.
            #    Minimum 5 second flight time enforced before landing is allowed.
            elif phase == "forward":
                april_pose    = None
                airborne_time = time.time() - takeoff_time

                if not sitl_mode and april is not None:
                    gray       = cv2.cvtColor(local_rgb, cv2.COLOR_BGR2GRAY)
                    detections = april.detect(
                        gray,
                        estimate_tag_pose=True,
                        camera_params=(FX, FY, CX, CY),
                        tag_size=TAG_SIZE_M
                    )
                    for tag in detections:
                        if tag.tag_id == TARGET_TAG_ID:
                            t          = tag.pose_t
                            april_pose = (float(t[0][0]), float(t[1][0]), float(t[2][0]))
                            break

                if april_pose is not None:
                    if airborne_time < MIN_FLIGHT_TIME_S:
                        remaining = MIN_FLIGHT_TIME_S - airborne_time
                        logger.info(f"[C2] Tag detected but min flight time not met — continuing forward {remaining:.1f}s")
                        controller.send_velocity(FORWARD_SPEED_MS, 0, 0)
                    else:
                        logger.info(f"[C2] AprilTag ID={TARGET_TAG_ID} detected — switching to TRACK")
                        last_tag_time = time.time()
                        phase = "track"
                else:
                    controller.send_velocity(FORWARD_SPEED_MS, 0, 0)
                    logger.debug(f"[C2] Searching for UGV — airborne {airborne_time:.1f}s")

            # 9. TRACK phase — chase AprilTag on moving UGV with PID velocity control.
            #    Identical tag lost handling to stationary_landing.py and mission_c1.py:
            #    coast on last velocity → ascend to widen FOV → emergency blind landing.
            elif phase == "track":
                april_pose = None
                if not sitl_mode and april is not None:
                    gray       = cv2.cvtColor(local_rgb, cv2.COLOR_BGR2GRAY)
                    detections = april.detect(
                        gray,
                        estimate_tag_pose=True,
                        camera_params=(FX, FY, CX, CY),
                        tag_size=TAG_SIZE_M
                    )
                    for tag in detections:
                        if tag.tag_id == TARGET_TAG_ID:
                            t          = tag.pose_t
                            april_pose = (float(t[0][0]), float(t[1][0]), float(t[2][0]))
                            break

                if april_pose is None:
                    time_lost = time.time() - last_tag_time

                    # Ground escape failsafe — dangerously low while blind, climb first
                    if is_escaping_ground or controller.prev_z < 0.2:
                        if not is_escaping_ground:
                            is_escaping_ground = True
                            escape_target_z    = controller.prev_z + 0.5
                            logger.warning(f"[C2] Dangerously low ({controller.prev_z:.2f}m) while blind — forcing 0.5m escape climb")
                        if controller.prev_z < escape_target_z:
                            controller.send_velocity(
                                controller.last_vx * COAST_BOOST,
                                controller.last_vy * COAST_BOOST,
                                -0.5
                            )
                        else:
                            logger.info("[C2] Ground escape complete — resuming track")
                            is_escaping_ground = False

                    # Low altitude land failsafe — if low and still blind, land now
                    elif controller.prev_z < 0.6:
                        logger.warning(f"[C2] Tag lost at low altitude ({controller.prev_z:.2f}m) — forced touchdown")
                        controller.smart_touchdown(timeout=3.0)
                        controller.disarm_motors()
                        wait_start = time.time()
                        phase = "wait"

                    # Coast — UGV is moving, keep up on predicted path
                    elif time_lost < HOVER_TIMEOUT_S:
                        logger.debug(f"[C2] Tag lost {time_lost:.1f}s — coasting on predicted path")
                        controller.coast_on_last_velocity(boost_multiplier=COAST_BOOST, vertical_velocity=0.0)

                    # Ascend — gain altitude to widen camera FOV
                    elif time_lost < SEARCH_TIMEOUT_S:
                        logger.warning(f"[C2] Tag lost {time_lost:.1f}s — ascending to widen FOV")
                        controller.coast_on_last_velocity(boost_multiplier=COAST_BOOST, vertical_velocity=-0.4)

                    # Give up — tag lost too long, land blind
                    else:
                        logger.error(f"[C2] Tag lost {time_lost:.1f}s — emergency blind landing")
                        controller.smart_touchdown(timeout=6.0)
                        controller.disarm_motors()
                        wait_start = time.time()
                        phase = "wait"

                else:
                    # Tag visible — reset timer and track
                    is_escaping_ground = False
                    last_tag_time      = time.time()
                    cam_x, cam_y, cam_z = april_pose
                    print(f"[INFO] Camera Frame | X={cam_x:.2f}, Y={cam_y:.2f}, Z={cam_z:.2f}")
                    body_x, body_y, body_z = controller.convert_camera_to_body_frame(cam_x, cam_y, cam_z)
                    print(f"[INFO] Body Frame | X={body_x:.2f}, Y={body_y:.2f}, Z={body_z:.2f}")

                    # Landing conditions — centered above tag and within threshold distance
                    if abs(body_x) < 1.0 and abs(body_y) < 1.0 and body_z < LANDING_THRESHOLD_M:
                        logger.info(f"[C2] Landing conditions met — body x={body_x:.2f} y={body_y:.2f} z={body_z:.2f}")
                        phase = "land"
                    else:
                        controller.adjust_velocity_and_send(body_x, body_y, body_z)

            # 10. LAND phase — smart_touchdown detects physical contact by monitoring
            #     vertical speed flatlining. Disarm immediately so drone sits as dead
            #     weight on the UGV while it continues toward the destination marker.
            elif phase == "land":
                logger.info("[C2] Phase: LAND — smart touchdown initiating")
                controller.smart_touchdown(timeout=8.0)
                logger.info("[C2] Touchdown complete — disarming")
                controller.disarm_motors()
                state.set_flight_mode(FlightMode.LAND)
                wait_start = time.time()
                phase = "wait"
                logger.info(f"[C2] Phase: WAIT — sitting on UGV for {WAIT_ON_UGV_S}s")

            # 11. WAIT phase — drone is disarmed on the moving UGV.
            #     UGV continues toward the ArUco marker destination.
            #     Time stops once UGV reaches destination and stops moving —
            #     we just wait 10 seconds and call it done regardless.
            elif phase == "wait":
                elapsed   = time.time() - wait_start
                remaining = WAIT_ON_UGV_S - elapsed
                if elapsed >= WAIT_ON_UGV_S:
                    logger.info("[C2] 10 seconds on UGV complete — challenge finished")
                    phase = "done"
                else:
                    if int(elapsed) % 5 == 0 and int(elapsed) > 0:
                        logger.info(f"[C2] On UGV — {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")
                    time.sleep(1.0)
                    continue

            elapsed_s = time.time() - t_start
            time.sleep(max(0.0, LOOP_PERIOD - elapsed_s))

    except KeyboardInterrupt:
        logger.warning("[C2] Interrupted — attempting safe landing")
        controller.smart_touchdown(timeout=8.0)
        controller.disarm_motors()

    finally:
        if shm_rgb is not None:
            shm_rgb.close()
        if shm_calib is not None:
            shm_calib.close()
        state.close()
        total_time = time.time() - takeoff_time
        logger.info(f"[C2] Process exiting — total flight time {total_time:.1f}s")