''' Main file to run entire UAV, now w/multiprocessing '''

# python main.py --mode scan

import time
import argparse
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import numpy as np
import depthai as dai
import cv2 as _cv2

# Multiprocessing-safe state
from core.state import (
    create_shared_state,
    cleanup_shared_state,
    UAVStateAccessor,
    FlightMode
)
from core.config import load_config
from core.log import get_logger

# VIO components
from vio_slam.vo_full_v3 import (
    VO_LK, LoopClosureORB,
    RGB_SOCKET, LEFT_SOCKET, RIGHT_SOCKET,
    ENABLE_LOOP, KEYFRAME_INTERVAL,
    FPS, W, H, IMU_HZ
)

# Vision
from vision.common.detectors.detector_manager import DetectorManager
from vision.common.video.camera_coordinate_transformer import CameraCoordinateTransformer

# Landing controller (used by lawnmower process only during testing)
from mission.pixhawk_controller.stationary_landing_controller import StationaryLandingController

# AprilTag — imported but process disabled during lawnmower testing
# from vision.common.detectors.april_detector.april_tag_detector import AprilTagDetector

# Lawnmower — functional API, reacts to marker_confirmed Event from run_vision
from mission.lawnmower_search import run_lawnmower_mission

# Telemetry
from telemetry.telemetry_logger import telemetry_logger

logger = get_logger(__name__)


# ===========================================================================
# Process 1: SLAM (VIO with loop closure)
# ===========================================================================

def run_slam(lock, marker_confirmed, ugv_signal, hover_reached, cfg):
    """
    Visual-inertial odometry process.

    Runs stereo depth + optical flow + loop closure.
    Writes position to shared memory every frame.
    This process never exits — runs for the entire flight.
    """
    logger.info("[SLAM] Process starting")

    state = UAVStateAccessor(lock, marker_confirmed, ugv_signal, hover_reached)

    try:
        with dai.Device() as device:
            with dai.Pipeline(device) as pipeline:
                cam_rgb   = pipeline.create(dai.node.Camera).build(RGB_SOCKET)
                cam_left  = pipeline.create(dai.node.Camera).build(LEFT_SOCKET)
                cam_right = pipeline.create(dai.node.Camera).build(RIGHT_SOCKET)

                stereo = pipeline.create(dai.node.StereoDepth)
                imu    = pipeline.create(dai.node.IMU)
                sync   = pipeline.create(dai.node.Sync)

                stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
                stereo.setLeftRightCheck(True)
                stereo.setSubpixel(True)
                stereo.setDepthAlign(
                    dai.StereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT
                )

                imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, IMU_HZ)
                imu.setBatchReportThreshold(1)
                imu.setMaxBatchReports(10)

                rgb_out   = cam_rgb.requestOutput(size=(W, H), fps=FPS, enableUndistortion=True)
                left_out  = cam_left.requestOutput(size=(W, H), fps=FPS)
                right_out = cam_right.requestOutput(size=(W, H), fps=FPS)

                left_out.link(stereo.left)
                right_out.link(stereo.right)

                rgb_out.link(sync.inputs["rgb"])
                stereo.rectifiedLeft.link(sync.inputs["left"])
                stereo.depth.link(sync.inputs["depth"])

                calib = device.getCalibration()
                K = np.array(
                    calib.getCameraIntrinsics(LEFT_SOCKET, W, H),
                    dtype=np.float64
                )

                vo   = VO_LK(K)
                loop = LoopClosureORB() if ENABLE_LOOP else None

                sync_q = sync.out.createOutputQueue()
                imu_q  = imu.out.createOutputQueue(maxSize=50, blocking=False)

                pipeline.start()

                frame_id = 0
                t0 = time.time()

                logger.info("[SLAM] Pipeline running")

                while pipeline.isRunning():
                    try:
                        for msg in imu_q.tryGetAll():
                            for pkt in msg.packets:
                                vo.update_imu(pkt.gyroscope.z)
                    except Exception:
                        pass

                    msg_group = sync_q.get()
                    if msg_group is None:
                        continue

                    gray     = msg_group["left"].getCvFrame()
                    depth_mm = msg_group["depth"].getFrame()

                    if gray is None or depth_mm is None:
                        continue

                    vo.process(gray, depth_mm)
                    pos, yaw_vis = vo.pose()

                    state.set_vio_position(
                        float(pos[0]),
                        float(pos[1]),
                        float(pos[2]),
                        float(yaw_vis)
                    )

                    if ENABLE_LOOP and loop and vo.status == "TRACKING":
                        rgb = msg_group["rgb"].getCvFrame()
                        if (frame_id % KEYFRAME_INTERVAL) == 0:
                            loop.add_keyframe(rgb, pos, frame_id, time.time() - t0)

                        info = loop.check_loop(rgb, pos, frame_id)
                        if info:
                            vo.apply_soft_correction(info["matched_pose"])
                            pos, yaw_vis = vo.pose()
                            state.set_vio_position(
                                float(pos[0]),
                                float(pos[1]),
                                float(pos[2]),
                                float(yaw_vis)
                            )

                    frame_id += 1

    finally:
        state.close()
        logger.info("[SLAM] Process exiting")


def save_marker_snapshot(frame, corners, ids, marker_id: int, log_timestamp: str):
    save_dir = Path("flight_logs/markers")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Draw bounding box and ID on a copy of the frame
    annotated = frame.copy()
    annotated = _cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

    filename = save_dir / f"marker_{marker_id}_{log_timestamp}.png"
    _cv2.imwrite(str(filename), annotated)

    logger.info(f"[VISION] Marker snapshot saved → {filename}")


def run_vision(lock, marker_confirmed, ugv_signal, hover_reached, cfg,
               log_timestamp: str):
    """
    ArUco marker detection process.

    Opens its own DepthAI pipeline on the RGB camera.
    Uses DetectorManager -> Cv2Detector for ArUco detection.
    When a valid marker is found:
        - Saves a color snapshot with bounding box to flight_logs/markers/
        - Writes pose to shared memory
        - Sets marker_confirmed (multiprocessing.Event) via set_aruco_pose()

    Continues sleeping after confirmation so process cleanup stays clean.

    log_timestamp is passed from main() so the snapshot filename matches
    the telemetry database for easy cross-referencing after flight.
    """
    logger.info("[VISION] Process starting")

    state = UAVStateAccessor(lock, marker_confirmed, ugv_signal, hover_reached)

    target_ids = cfg.aruco.target_marker_id
    if not isinstance(target_ids, list):
        target_ids = [target_ids]

    # Instantiate inside the process — not safe to pickle across spawn
    detector    = DetectorManager(cfg.detector).get_detector()
    transformer = CameraCoordinateTransformer(cfg.video)

    try:
        with dai.Device() as device:
            with dai.Pipeline(device) as pipeline:
                cam_rgb = pipeline.create(dai.node.Camera).build(
                    dai.CameraBoardSocket.CAM_A
                )
                rgb_out = cam_rgb.requestOutput(
                    size=(640, 480),
                    fps=cfg.camera.color_fps
                )
                q_rgb = rgb_out.createOutputQueue(maxSize=4, blocking=False)

                pipeline.start()

                logger.info(f"[VISION] Scanning for ArUco IDs: {target_ids}")

                while pipeline.isRunning():
                    if marker_confirmed.is_set():
                        time.sleep(1.0)
                        continue

                    frame_msg = q_rgb.tryGet()
                    if frame_msg is None:
                        time.sleep(0.005)
                        continue

                    frame = frame_msg.getCvFrame()

                    corners, ids, _ = detector.detect(frame)
                    if ids is None:
                        continue

                    flat_ids = ids.flatten().tolist()
                    matched  = [i for i in flat_ids if i in target_ids]
                    if not matched:
                        continue

                    center_x = (
                        corners[0][0][0][0] +
                        (corners[0][0][2][0] - corners[0][0][0][0]) / 2
                    )
                    center_y = (
                        corners[0][0][0][1] +
                        (corners[0][0][2][1] - corners[0][0][0][1]) / 2
                    )
                    x, y, z = transformer.transform((center_x, center_y), 10)

                    # Save annotated color snapshot before setting the event
                    save_marker_snapshot(frame, corners, ids, matched[0], log_timestamp)

                    # Writes to shared memory and sets marker_confirmed Event
                    state.set_aruco_pose(x, y, z, matched[0])
                    logger.info(
                        f"[VISION] ArUco ID={matched[0]} confirmed — "
                        f"x={x:.2f} y={y:.2f} z={z:.2f}"
                    )

    finally:
        state.close()
        logger.info("[VISION] Process exiting")


def run_lawnmower(lock, marker_confirmed, ugv_signal, hover_reached, cfg):
    """
    Lawnmower search pattern process.

    Holds the only MAVLink connection during testing (serial port is
    exclusive — run_landing is disabled below to avoid port conflict).

    Flies boustrophedon pattern reading VIO from shared memory.
    Reacts to marker_confirmed (set by run_vision) by:
        1. Aborting the sweep immediately
        2. Flying to the VO-snapshotted marker position
        3. Descending to confirm altitude
        4. Calling StationaryLandingController.stationary_landing()
        5. Calling disarm_motors()

    TODO: When re-enabling AprilTag precision landing (run_landing):
        - Comment out stationary_landing() + disarm_motors() in
          fly_to_marker_and_land() inside lawnmower_search.py
        - Uncomment p_landing in main() below
        - Move StationaryLandingController instantiation to run_landing
          so only that process holds the serial port after marker confirm
    """
    logger.info("[LAWNMOWER] Process starting")

    state = UAVStateAccessor(lock, marker_confirmed, ugv_signal, hover_reached)

    # Sole MAVLink connection during testing
    controller = StationaryLandingController(
        cfg.pixhawk.connection_string,
        cfg.pixhawk.baud_rate
    )

    class StateVOAdapter:
        """Adapts shared-memory VIO state to the .pose() interface."""
        def __init__(self, state_accessor):
            self._state = state_accessor

        @property
        def status(self):
            (x, y, z, yaw), timestamp = self._state.get_vio_position()
            return "TRACKING" if timestamp > 0 else "INIT"

        def pose(self):
            (x, y, z, yaw), _ = self._state.get_vio_position()
            # lawnmower_search expects [x_right, y_down, z_forward]
            return np.array([x, y, z]), yaw

    vo_adapter = StateVOAdapter(state)

    def on_marker_confirmed(north, east):
        """Log final VO position at landing point for post-flight review."""
        logger.info(
            f"[LAWNMOWER] Landing at VO position N={north:.2f} E={east:.2f}"
        )

    try:
        valid_ids = (
            cfg.aruco.target_marker_id
            if isinstance(cfg.aruco.target_marker_id, list)
            else [cfg.aruco.target_marker_id]
        )

        # Blocking — returns when landed or field exhausted
        mission_state = run_lawnmower_mission(
            mav_master=controller.master,
            vo=vo_adapter,
            valid_ids=valid_ids,
            marker_confirmed=marker_confirmed,
            controller=controller,
        )

        if mission_state["valid_marker_confirmed"]:
            pos = mission_state["confirmed_marker_position"]
            logger.info(
                f"[LAWNMOWER] Mission complete — "
                f"landed at N={pos[0]:.2f} E={pos[1]:.2f}"
            )
        else:
            logger.warning("[LAWNMOWER] Mission complete — marker NOT found")

    finally:
        state.close()
        logger.info("[LAWNMOWER] Process exiting")


# ===========================================================================
# Process 4: AprilTag Precision Landing
# DISABLED during lawnmower testing — re-enable once lawnmower is validated.
# When re-enabling:
#   1. Uncomment this function and p_landing in main()
#   2. Comment out stationary_landing()/disarm_motors() in lawnmower_search.py
#   3. Move StationaryLandingController to this process only
# ===========================================================================
'''
 def run_landing(lock, marker_confirmed, ugv_signal, hover_reached, cfg):
     """
     AprilTag precision landing process.
     Blocks on marker_confirmed then begins precision landing loop.
     """
     logger.info("[LANDING] Process starting")

     state = UAVStateAccessor(lock, marker_confirmed, ugv_signal, hover_reached)

     logger.info("[LANDING] Waiting for ArUco marker confirmation...")
     marker_confirmed.wait()

     logger.info("[LANDING] Marker confirmed — starting AprilTag sequence")
     state.set_flight_mode(FlightMode.LAND)

     controller = StationaryLandingController(
         cfg.pixhawk.connection_string,
         cfg.pixhawk.baud_rate
     )

     try:
         with dai.Device() as device:
             calibration        = device.getCalibration()
             april_tag_detector = AprilTagDetector(calibration)

             with dai.Pipeline(device) as pipeline:
                 cam_rgb = pipeline.create(dai.node.Camera).build(
                     dai.CameraBoardSocket.CAM_A
                 )
                 rgb_out = cam_rgb.requestOutput(
                     size=(300, 300),
                     type=dai.ImgFrame.Type.NV12,
                     fps=30
                 )
                 q_rgb = rgb_out.createOutputQueue(maxSize=4, blocking=False)
                 pipeline.start()

                 last_tag_time  = time.time()
                 HOVER_TIMEOUT  = 4.0
                 SEARCH_TIMEOUT = 7.0

                 logger.info("[LANDING] AprilTag tracking loop running")

                 while pipeline.isRunning():
                     in_rgb = q_rgb.get()
                     if in_rgb is None:
                         continue

                     frame = in_rgb.getCvFrame()
                     pose  = april_tag_detector.get_tag_pose(frame)

                     if pose is None:
                         time_lost = time.time() - last_tag_time
                         if time_lost < HOVER_TIMEOUT:
                             controller.send_velocity(0, 0, 0)
                         elif time_lost < SEARCH_TIMEOUT:
                             logger.warning(f"[LANDING] Tag lost {time_lost:.1f}s — ascending")
                             controller.send_velocity(0, 0, -0.2)
                         else:
                             logger.error("[LANDING] Tag lost >7s — blind descent")
                             controller.send_velocity(0, 0, 0.3)
                         continue

                     last_tag_time = time.time()
                     cam_x, cam_y, cam_z    = pose
                     body_x, body_y, body_z = controller.convert_camera_to_body_frame(
                         cam_x, cam_y, cam_z
                     )
                     logger.info(
                         f"[LANDING] body x={body_x:.2f} "
                         f"y={body_y:.2f} z={body_z:.2f}"
                     )
                     if body_z < cfg.pixhawk.landing_threshold:
                         logger.info("[LANDING] Threshold reached — landing")
                         controller.stationary_landing()
                         time.sleep(5)
                         controller.disarm_motors()
                         break
                     controller.adjust_velocity_and_send(body_x, body_y, body_z)

     finally:
         state.close()
         logger.info("[LANDING] Process exiting")
'''




def main():
    parser = argparse.ArgumentParser(
        description="UAV autonomous mission — multiprocessing"
    )
    parser.add_argument("--mode", choices=["scan", "land"], default="scan")
    args = parser.parse_args()

    cfg = load_config(mode=args.mode)

    logger.info(f"Starting UAV system in mode: {cfg.mode}")
    logger.info("Using multiprocessing architecture")

    state_dict       = create_shared_state()
    lock             = state_dict["lock"]
    marker_confirmed = state_dict["marker_confirmed"]
    ugv_signal       = state_dict["ugv_signal"]
    hover_reached    = state_dict["hover_reached"]

    state = UAVStateAccessor(lock, marker_confirmed, ugv_signal, hover_reached)

    Path("flight_logs").mkdir(exist_ok=True)
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = f"flight_logs/flight_{log_timestamp}.db"

    # Arm and take off from main process before spawning children.
    # Main process connects briefly then closes — lawnmower process
    # opens its own connection after spawn.
    logger.info("[MAIN] Connecting to Pixhawk for arm/takeoff")
    controller = StationaryLandingController(
        cfg.pixhawk.connection_string,
        cfg.pixhawk.baud_rate
    )

    logger.info("[MAIN] Arming and taking off")
    controller.change_flight_mode("GUIDED")
    controller.arm_motors()
    controller.takeoff_to_altitude(cfg.pixhawk.hover_altitude_m)

    # Close main process MAVLink connection before spawning lawnmower
    # so the serial port is free for the lawnmower process to claim
    del controller

    state.set_flight_mode(FlightMode.SCAN)
    logger.info("[MAIN] Takeoff complete — spawning processes")

    processes = []

    # Process 1: SLAM
    p_slam = mp.Process(
        target=run_slam,
        args=(lock, marker_confirmed, ugv_signal, hover_reached, cfg),
        name="slam"
    )
    processes.append(p_slam)

    # Process 2: Vision (ArUco — sets marker_confirmed, saves snapshot)
    p_vision = mp.Process(
        target=run_vision,
        args=(lock, marker_confirmed, ugv_signal, hover_reached, cfg,
              log_timestamp),
        name="vision"
    )
    processes.append(p_vision)

    # Process 3: Lawnmower (flight + landing — sole MAVLink owner during testing)
    p_lawnmower = mp.Process(
        target=run_lawnmower,
        args=(lock, marker_confirmed, ugv_signal, hover_reached, cfg),
        name="lawnmower"
    )
    processes.append(p_lawnmower)

    # Process 4: AprilTag precision landing
    # DISABLED during lawnmower testing — see run_landing() above
    # p_landing = mp.Process(
    #     target=run_landing,
    #     args=(lock, marker_confirmed, ugv_signal, hover_reached, cfg),
    #     name="landing"
    # )
    # processes.append(p_landing)

    # Process 5: Telemetry
    p_telemetry = mp.Process(
        target=telemetry_logger,
        args=(lock, db_path, cfg.pixhawk.connection_string, cfg.pixhawk.baud_rate),
        name="telemetry"
    )
    processes.append(p_telemetry)

    for p in processes:
        p.start()
        logger.info(f"[MAIN] Started process: {p.name}")

    logger.info(f"[MAIN] All processes running — logging to {db_path}")

    # Mission complete when lawnmower exits (landing is self-contained)
    try:
        while True:
            mode = state.get_flight_mode()
            (vio_x, vio_y, vio_z, vio_yaw), vio_ts = state.get_vio_position()
            (aruco_pos, marker_id) = state.get_aruco_pose()

            vio_str    = f"{vio_x:.2f},{vio_y:.2f},{vio_z:.2f}" if vio_ts > 0 else "None"
            marker_str = f"ID:{marker_id}" if marker_id else "None"

            logger.info(
                f"[STATUS] mode={mode.name} | "
                f"vio={vio_str} | "
                f"marker={marker_str}"
            )

            # During testing: mission complete when lawnmower process exits
            if not p_lawnmower.is_alive():
                logger.info("[MAIN] Lawnmower finished — mission complete")
                break

            time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info("[MAIN] Keyboard interrupt — shutting down")

    finally:
        logger.info("[MAIN] Terminating all processes")
        for p in processes:
            if p.is_alive():
                p.terminate()

        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                logger.warning(f"[MAIN] Force killing {p.name}")
                p.kill()

        state.close()
        cleanup_shared_state(state_dict)

        logger.info(f"[MAIN] Shutdown complete. Telemetry saved to {db_path}")
        logger.info("[MAIN] Done. Yabadabadoo!")


if __name__ == "__main__":
    # CRITICAL: must use 'spawn' on Raspberry Pi
    # 'fork' causes issues with DepthAI and OpenCV
    mp.set_start_method('spawn', force=True)
    main()