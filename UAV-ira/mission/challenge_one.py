''' Challenge 1 — UAV autonomously launches, UGV system begins traveling in straight line
UAV autonomously lands on the moving UGV system and continues traveling for thirty
seconds '''

# python main.py --mode scan --planner c1

import time
import numpy as np
import cv2
from multiprocessing import shared_memory
from pupil_apriltags import Detector as AprilDetector

from core.state import UAVStateAccessor, FlightMode
from core.log import get_logger
from landing.pixhawk_controller.stationary_landing_controller import StationaryLandingController

logger = get_logger(__name__)

TAKEOFF_ALTITUDE_M = 3.0    # above 4 foot minimum (1.22m)
MIN_FLIGHT_TIME_S  = 5.0    # challenge requires minimum 5 seconds airborne
MISSION_TIMEOUT_S  = 420.0  # 7 minutes max to land on UGV
WAIT_ON_UGV_S = 30.0   # sit on UGV after landing without separating

# Forward search phase
FORWARD_SPEED_MS = 0.5  # m/s in body frame, heading at takeoff does not matter

# AprilTag detection
# switched to a smaller 5.5 by 5.5 aruco marker
TARGET_TAG_ID = 67
TAG_SIZE_M = 0.14

# Tag tracking and landing
LANDING_THRESHOLD_M = 0.4   # z distance to UGV surface before touchdown
HOVER_TIMEOUT_S = 3.0   # coast on last velocity when tag lost
SEARCH_TIMEOUT_S = 7.0   # ascend to widen FOV if still lost
COAST_BOOST = 1.3   # velocity multiplier when coasting blind

LOOP_HZ= 30
LOOP_PERIOD = 1.0 / LOOP_HZ


def run_mission(lock, marker_confirmed, ugv_signal, hover_reached,cfg, log_timestamp, uncertain_pos, planner, rgb_frame_mutex):
    '''Challenge 1 mission process.

    Phase sequence:
        forward → fly in body frame until AprilTag enters frame
        track   → chase tag with PID velocity control
        land    → smart_touchdown once centered above tag
        wait    → disarmed on moving UGV for 30 seconds
        done    → mission complete
    '''
    logger.info("[C1] Process starting")

    W, H  = cfg.camera.width, cfg.camera.height
    state = UAVStateAccessor(lock, marker_confirmed, ugv_signal, hover_reached)

    controller = StationaryLandingController(
        cfg.pixhawk.connection_string,
        cfg.pixhawk.baud_rate
    )

    # 1. Arm and take off. UGV is stationary at this point.
    #    Challenge says UGV begins moving AFTER the UAV launches.
    logger.info("[C1] Arming and taking off")
    controller.change_flight_mode("GUIDED")
    controller.arm_motors()
    controller.takeoff_to_altitude(TAKEOFF_ALTITUDE_M)
    state.set_flight_mode(FlightMode.SCAN)
    takeoff_time  = time.time()
    mission_start = time.time()
    logger.info(f"[C1] Airborne at {TAKEOFF_ALTITUDE_M}m — minimum flight time {MIN_FLIGHT_TIME_S}s starts now")

    # 2. Connect to shared memory for RGB frames and camera calibration.
    #    Broadcaster owns the OAK-D — we only read from shared memory here.
    #    In SITL mode neither exists so we skip detection entirely.
    sitl_mode  = not cfg.pixhawk.connection_string.startswith("/dev")
    shm_rgb = None
    shm_calib  = None
    shared_rgb = None
    local_rgb  = np.zeros((H, W, 3), dtype=np.uint8)
    april = None
    FX = FY = CX = CY = 0.0

    if not sitl_mode:
        try:
            shm_rgb = shared_memory.SharedMemory(name="oak_rgb")
            shared_rgb = np.ndarray((H, W, 3), dtype=np.uint8, buffer=shm_rgb.buf)
            # Calibration is written once by broadcaster at startup
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
            logger.info(f"[C1] AprilTag detector ready — fx={FX:.2f} fy={FY:.2f} cx={CX:.2f} cy={CY:.2f}")
        except Exception as e:
            logger.warning(f"[C1] Camera shared memory unavailable: {e} — treating as SITL")
            sitl_mode = True

    if sitl_mode:
        logger.info("[C1] SITL mode — camera disabled, forward phase will run until timeout")

    # 3. Phase state machine
    phase = "forward"
    last_tag_time = time.time()
    is_escaping_ground = False
    escape_target_z = 0.0
    wait_start = 0.0

    logger.info("[C1] Phase: FORWARD — flying forward until AprilTag detected")

    try:
        while phase != "done":
            t_start = time.time()

            # Global 7 min limit, land no matter what
            if time.time() - mission_start > MISSION_TIMEOUT_S:
                logger.error("[C1] Mission timeout (7 min) — emergency landing")
                controller.smart_touchdown(timeout=8.0)
                controller.disarm_motors()
                break

            # 4. Read the latest RGB frame from shared memory and run AprilTag detection.
            #    This runs every loop tick regardless of phase so we never miss the tag.
            pose = None
            if not sitl_mode and shared_rgb is not None:
                with rgb_frame_mutex:
                    np.copyto(local_rgb, shared_rgb)
                gray = cv2.cvtColor(local_rgb, cv2.COLOR_BGR2GRAY)
                detections = april.detect(
                    gray,
                    estimate_tag_pose=True,
                    camera_params=(FX, FY, CX, CY),
                    tag_size=TAG_SIZE_M
                )
                for tag in detections:
                    if tag.tag_id == TARGET_TAG_ID:
                        t = tag.pose_t
                        pose = (float(t[0][0]), float(t[1][0]), float(t[2][0]))
                        break

            # 5. FORWARD phase — fly w/body frame until tag appears(vx=forward)
            if phase == "forward":
                airborne_time = time.time() - takeoff_time
                if pose is not None:
                    if airborne_time < MIN_FLIGHT_TIME_S:
                        remaining = MIN_FLIGHT_TIME_S - airborne_time
                        logger.info(f"[C1] Tag detected but min flight time not met — continuing forward {remaining:.1f}s")
                        controller.send_velocity(FORWARD_SPEED_MS, 0, 0)
                    else:
                        logger.info(f"[C1] AprilTag ID={TARGET_TAG_ID} detected after {airborne_time:.1f}s — switching to TRACK")
                        last_tag_time = time.time()
                        phase = "track"
                else:
                    controller.send_velocity(FORWARD_SPEED_MS, 0, 0)
                    logger.debug(f"[C1] Searching — airborne {airborne_time:.1f}s | flying forward at {FORWARD_SPEED_MS}m/s")

            # 6. TRACK phase — chase the AprilTag with PID velocity control.
            #    Tag lost logic mirrors stationary_landing.py exactly:
            #    coast → ascend to widen FOV → emergency blind landing.
            elif phase == "track":
                if pose is None:
                    time_lost = time.time() - last_tag_time

                    # Ground escape failsafe — if dangerously low while blind, climb first
                    if is_escaping_ground or controller.prev_z < 0.2:
                        if not is_escaping_ground:
                            is_escaping_ground = True
                            escape_target_z    = controller.prev_z + 0.5
                            logger.warning(f"[C1] Dangerously low ({controller.prev_z:.2f}m) while blind — forcing 0.5m escape climb")
                        if controller.prev_z < escape_target_z:
                            controller.send_velocity(
                                controller.last_vx * COAST_BOOST,
                                controller.last_vy * COAST_BOOST,
                                -0.5
                            )
                        else:
                            logger.info("[C1] Ground escape complete — resuming track")
                            is_escaping_ground = False

                    # Low altitude land failsafe — if low and still blind, land now
                    elif controller.prev_z < 0.6:
                        logger.warning(f"[C1] Tag lost at low altitude ({controller.prev_z:.2f}m) — forced touchdown")
                        controller.smart_touchdown(timeout=3.0)
                        controller.disarm_motors()
                        wait_start = time.time()
                        phase = "wait"

                    # Coast, UGV is still moving, keep up on predicted path
                    elif time_lost < HOVER_TIMEOUT_S:
                        logger.debug(f"[C1] Tag lost {time_lost:.1f}s — coasting on predicted path")
                        controller.coast_on_last_velocity(boost_multiplier=COAST_BOOST, vertical_velocity=0.0)

                    # increase altitude to widen camera FOV
                    elif time_lost < SEARCH_TIMEOUT_S:
                        logger.warning(f"[C1] Tag lost {time_lost:.1f}s — ascending to widen FOV")
                        controller.coast_on_last_velocity(boost_multiplier=COAST_BOOST, vertical_velocity=-0.4)

                    # Tag lost too long, land blind
                    else:
                        logger.error(f"[C1] Tag lost {time_lost:.1f}s — emergency blind landing")
                        controller.smart_touchdown(timeout=6.0)
                        controller.disarm_motors()
                        wait_start = time.time()
                        phase = "wait"

                else:
                    # Tag visible — reset timer and track
                    is_escaping_ground = False
                    last_tag_time      = time.time()
                    cam_x, cam_y, cam_z = pose
                    print(f"[INFO] Camera Frame | X={cam_x:.2f}, Y={cam_y:.2f}, Z={cam_z:.2f}")
                    body_x, body_y, body_z = controller.convert_camera_to_body_frame(cam_x, cam_y, cam_z)
                    print(f"[INFO] Body Frame | X={body_x:.2f}, Y={body_y:.2f}, Z={body_z:.2f}")

                    # Landing conditions — centered above tag and within threshold distance
                    if abs(body_x) < 1.0 and abs(body_y) < 1.0 and body_z < LANDING_THRESHOLD_M:
                        logger.info(f"[C1] Landing conditions met — body x={body_x:.2f} y={body_y:.2f} z={body_z:.2f}")
                        phase = "land"
                    else:
                        controller.adjust_velocity_and_send(body_x, body_y, body_z)

            # 7. LAND phase — smart_touchdown detects physical contact by monitoring
            #    vertical speed flatlining. Disarm immediately after so the drone
            #    sits as dead weight on the UGV platform.
            elif phase == "land":
                logger.info("[C1] Phase: LAND — smart touchdown initiating")
                controller.smart_touchdown(timeout=8.0)
                logger.info("[C1] Touchdown complete — disarming")
                controller.disarm_motors()
                state.set_flight_mode(FlightMode.LAND)
                wait_start = time.time()
                phase = "wait"
                logger.info(f"[C1] Phase: WAIT — sitting on UGV for {WAIT_ON_UGV_S}s")

            # 8. WAIT phase — drone is disarmed and sitting on the moving UGV.
            #    Nothing to do except count down 30 seconds. Logs every 5 seconds
            #    so you can confirm the timer is running during the challenge.
            elif phase == "wait":
                elapsed   = time.time() - wait_start
                remaining = WAIT_ON_UGV_S - elapsed
                if elapsed >= WAIT_ON_UGV_S:
                    logger.info("[C1] 30 seconds on UGV complete — challenge finished")
                    phase = "done"
                else:
                    if int(elapsed) % 5 == 0 and int(elapsed) > 0:
                        logger.info(f"[C1] On UGV — {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")
                    time.sleep(1.0)
                    continue

            elapsed_s = time.time() - t_start
            time.sleep(max(0.0, LOOP_PERIOD - elapsed_s))

    except KeyboardInterrupt:
        logger.warning("[C1] Interrupted — attempting safe landing")
        controller.smart_touchdown(timeout=8.0)
        controller.disarm_motors()

    finally:
        if shm_rgb is not None:
            shm_rgb.close()
        if shm_calib is not None:
            shm_calib.close()
        state.close()
        total_time = time.time() - mission_start
        logger.info(f"[C1] Process exiting — total mission time {total_time:.1f}s")