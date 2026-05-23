''' Main file to run entire UAV, now w/multiprocessing '''

# python main.py --mode scan

import time
import argparse
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import numpy as np
import depthai as dai

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
#from vision.common.detectors.detector_manager import DetectorManager
from vision.common.detectors.opencv_helpers import Cv2Detector
detector = Cv2Detector()
from vision.common.video.camera_coordinate_transformer import CameraCoordinateTransformer

# Landing
from vision.common.detectors.april_detector.april_tag_detector import AprilTagDetector
from landing.pixhawk_controller.stationary_landing_controller import StationaryLandingController

# Lawnmower (uses threading internally, but runs in its own process)
from landing.lawnmower import LawnmowerMissionRunner

# Telemetry
from telemetry.telemetry_logger import telemetry_logger

logger = get_logger(__name__)


# =====
# Process 1: SLAM (VIO with loop closure)
# =====

def run_slam(lock, marker_confirmed, ugv_signal, hover_reached, cfg):
    """
    Visual-inertial odometry process.
    
    Runs stereo depth + optical flow + loop closure.
    Writes position to shared memory every frame.
    
    This process never exits — runs entire flight.
    """
    logger.info("[SLAM] Process starting")
    
    # Each process creates its own state accessor
    state = UAVStateAccessor(lock, marker_confirmed, ugv_signal, hover_reached)
    
    try:
        with dai.Device() as device:
            with dai.Pipeline(device) as pipeline:
                # Build camera nodes
                cam_rgb = pipeline.create(dai.node.Camera).build(RGB_SOCKET)
                cam_left = pipeline.create(dai.node.Camera).build(LEFT_SOCKET)
                cam_right = pipeline.create(dai.node.Camera).build(RIGHT_SOCKET)
                
                stereo = pipeline.create(dai.node.StereoDepth)
                imu = pipeline.create(dai.node.IMU)
                sync = pipeline.create(dai.node.Sync)
                
                stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
                stereo.setLeftRightCheck(True)
                stereo.setSubpixel(True)
                stereo.setDepthAlign(
                    dai.StereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT
                )
                
                imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, IMU_HZ)
                imu.setBatchReportThreshold(1)
                imu.setMaxBatchReports(10)
                
                rgb_out = cam_rgb.requestOutput(size=(W, H), fps=FPS, enableUndistortion=True)
                left_out = cam_left.requestOutput(size=(W, H), fps=FPS)
                right_out = cam_right.requestOutput(size=(W, H), fps=FPS)
                
                left_out.link(stereo.left)
                right_out.link(stereo.right)
                
                rgb_out.link(sync.inputs["rgb"])
                stereo.rectifiedLeft.link(sync.inputs["left"])
                stereo.depth.link(sync.inputs["depth"])
                
                # Camera intrinsics
                calib = device.getCalibration()
                K = np.array(
                    calib.getCameraIntrinsics(LEFT_SOCKET, W, H),
                    dtype=np.float64
                )
                
                vo = VO_LK(K)
                loop = LoopClosureORB() if ENABLE_LOOP else None
                
                sync_q = sync.out.createOutputQueue()
                imu_q = imu.out.createOutputQueue(maxSize=50, blocking=False)
                
                pipeline.start()
                
                frame_id = 0
                t0 = time.time()
                
                logger.info("[SLAM] Pipeline running")
                
                while pipeline.isRunning():
                    # Update IMU
                    try:
                        for msg in imu_q.tryGetAll():
                            for pkt in msg.packets:
                                vo.update_imu(pkt.gyroscope.z)
                    except Exception:
                        pass
                    
                    # Get synced frames
                    msg_group = sync_q.get()
                    if msg_group is None:
                        continue
                    
                    gray = msg_group["left"].getCvFrame()
                    depth_mm = msg_group["depth"].getFrame()
                    
                    if gray is None or depth_mm is None:
                        continue
                    
                    # Run VIO
                    vo.process(gray, depth_mm)
                    pos, yaw_vis = vo.pose()
                    
                    # Write to shared memory
                    state.set_vio_position(
                        float(pos[0]),
                        float(pos[1]),
                        float(pos[2]),
                        float(yaw_vis)
                    )
                    
                    # Loop closure
                    if ENABLE_LOOP and loop and vo.status == "TRACKING":
                        rgb = msg_group["rgb"].getCvFrame()
                        if (frame_id % KEYFRAME_INTERVAL) == 0:
                            loop.add_keyframe(rgb, pos, frame_id, time.time() - t0)
                        
                        info = loop.check_loop(rgb, pos, frame_id)
                        if info:
                            vo.apply_soft_correction(info["matched_pose"])
                            # Re-read corrected pose
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


# =====
# Process 2: Vision (ArUco detection)
# =====

def run_vision(lock, marker_confirmed, ugv_signal, hover_reached, cfg):
    """
    ArUco marker detection process.
    
    Scans for target marker IDs.
    When found, writes pose to shared memory and sets marker_confirmed Event.
    Then sleeps — doesn't exit so process cleanup is clean.
    """
    logger.info("[VISION] Process starting")
    
    state = UAVStateAccessor(lock, marker_confirmed, ugv_signal, hover_reached)
    
    try:
        with dai.Device() as device:
            with dai.Pipeline(device) as pipeline:
                cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
                rgb_out = cam_rgb.requestOutput(size=(640, 480), fps=cfg.camera.color_fps)
                q_rgb = rgb_out.createOutputQueue(maxSize=4, blocking=False)
                
                detector = DetectorManager(cfg.detector).get_detector()
                transformer = CameraCoordinateTransformer(cfg.video)
                
                target_ids = cfg.aruco.target_marker_id
                if not isinstance(target_ids, list):
                    target_ids = [target_ids]
                
                pipeline.start()
                
                logger.info(f"[VISION] Scanning for ArUco IDs: {target_ids}")
                
                while pipeline.isRunning():
                    # Stop scanning once marker confirmed
                    if marker_confirmed.is_set():
                        time.sleep(1.0)
                        continue
                    
                    frame_msg = q_rgb.tryGet()
                    if frame_msg is None:
                        time.sleep(0.005)
                        continue
                    
                    frame = frame_msg.getCvFrame()
                    
                    corners, ids, _ = detector.detect(frame, True)
                    if ids is None:
                        continue
                    
                    flat_ids = ids.flatten().tolist()
                    matched = [i for i in flat_ids if i in target_ids]
                    if not matched:
                        continue
                    
                    # Found valid marker — compute pose
                    center_x = corners[0][0][0][0] + (corners[0][0][2][0] - corners[0][0][0][0]) / 2
                    center_y = corners[0][0][0][1] + (corners[0][0][2][1] - corners[0][0][0][1]) / 2
                    x, y, z = transformer.transform((center_x, center_y), 10)
                    
                    state.set_aruco_pose(x, y, z, matched[0])
                    logger.info(f"[VISION] ✓ ArUco marker {matched[0]} confirmed at x={x:.2f} y={y:.2f} z={z:.2f}")
                    
                    # marker_confirmed.set() is called inside set_aruco_pose
    
    finally:
        state.close()
        logger.info("[VISION] Process exiting")


# =====
# Process 3: Lawnmower Search
# =====

def run_lawnmower(lock, marker_confirmed, ugv_signal, hover_reached, cfg):
    """
    Lawnmower search pattern process.
    
    Flies boustrophedon pattern reading VIO position from shared memory.
    Stops immediately when marker_confirmed Event is set.
    """
    logger.info("[LAWNMOWER] Process starting")
    
    state = UAVStateAccessor(lock, marker_confirmed, ugv_signal, hover_reached)
    
    # Each process creates its own MAVLink connection
    controller = StationaryLandingController(
        cfg.pixhawk.connection_string,
        cfg.pixhawk.baud_rate
    )

    
    # Adapter: LawnmowerMissionRunner expects a VO object with .pose() method


    # only enables gyroscope
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

    # camera intrisics
    calib = device.getCalibration()
    K = np.array(
        calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, W, H),
        dtype=np.float64
    )
    # Our class using lucas kanade method & closing loop w/ORB
    vo = VO_LK(K)
    loop = LoopClosureORB() if ENABLE_LOOP else None

    sync_q = sync.out.createOutputQueue()
    imu_q  = imu.out.createOutputQueue(maxSize=50, blocking=False)

    frame_id = 0
    t0 = time.time()

    logger.info("SLAM thread running")
    while pipeline.isRunning():
        # update IMU
        try:
            # msg represents a batch of IMU readings, which we loop through and feed to VO one by one
            # imu readings should look like this: [timestamp, gyro_x, gyro_y, gyro_z]?
            for msg in imu_q.tryGetAll():
                for pkt in msg.packets:
                    vo.update_imu(pkt.gyroscope.z)
        except Exception:
            pass

        # grabs next synced frame
        msg_group = sync_q.get()
        if msg_group is None:
            continue

        # gray is used for VO
        # rgb is for orb feature extaction
        gray= msg_group["left"].getCvFrame()
        depth_mm = msg_group["depth"].getFrame()
        if gray is None or depth_mm is None:
            continue

        vo.process(gray, depth_mm)
        pos, yaw_vis = vo.pose()

        # Write to shared memory for telemetry logger
        with lock:
            shared_vio[0] = float(pos[0])
            shared_vio[1] = float(pos[1])
            shared_vio[2] = float(pos[2])
            shared_vio[3] = float(yaw_vis)
            state.set_vio_position(Pose3D(
                    x=float(pos[0]),
                    y=float(pos[1]),
                    z=float(pos[2]),
                    yaw=yaw_vis
                ))

        if ENABLE_LOOP and loop and vo.status == "TRACKING":
            rgb = msg_group["rgb"].getCvFrame()
            if (frame_id % KEYFRAME_INTERVAL) == 0:
                loop.add_keyframe(rgb, pos, frame_id, time.time() - t0)
            info = loop.check_loop(rgb, pos, frame_id)
            if info:
                vo.apply_soft_correction(info["matched_pose"])

        frame_id += 1
        # got rid of visualization for now
        # we can add back later for testing if needed





#  Vision thread, wraps AutoUav logic from auto_uav.py with same dai pipeline
def run_vision(pipeline: dai.Pipeline, state: UAVState, cfg):
    """ Runs ArUco detection on the RGB stream, then sets state.marker_confirmed when the target marker is found."""

     # NOTE: CAM_A is already claimed by SLAM's rgb_out.
    # Vision reuses that same queue rather than creating a second node.
    # If your pipeline build order causes a conflict, create a second
    # requestOutput at a lower resolution here instead.

    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgb_out = cam_rgb.requestOutput(size=(640, 480), fps=cfg.camera.color_fps)
    q_rgb   = rgb_out.createOutputQueue(maxSize=4, blocking=False)

    detector   = DetectorManager(cfg.detector).get_detector()
    transformer = CameraCoordinateTransformer(cfg.video)
    segmenter  = FieldObstacleSegmenter(SegmenterConfig(process_interval_sec=0.5))

    # ex: [3, 7]
    target_ids = cfg.aruco.target_marker_id

    logger.info("Vision thread running")
    while pipeline.isRunning():
        # once we scan the correct marker, we can stop this thread
        if state.marker_confirmed.is_set():
            time.sleep(1.0)
            continue

        frame_msg = q_rgb.tryGet()
        if frame_msg is None:
            time.sleep(0.005)
            continue

        frame = frame_msg.getCvFrame()

        # Segmentation (obstacle awareness)
        if segmenter.should_process_now():
            _, grid = segmenter.process(frame)
            logger.debug(f"[SEG] occupied: {int(grid.sum())}/{grid.size}")

        # ArUco detection
        corners, ids, _ = detector.detect(frame, True)
        if ids is None:
            continue

        flat_ids = ids.flatten().tolist()
        matched  = [i for i in flat_ids if i in target_ids]
        if not matched:
            continue

        # Compute pose of first matched marker
        center_x = corners[0][0][0][0] + (corners[0][0][2][0] - corners[0][0][0][0]) / 2
        center_y = corners[0][0][0][1] + (corners[0][0][2][1] - corners[0][0][0][1]) / 2
        x, y, z  = transformer.transform((center_x, center_y), 10)

        state.set_aruco_pose(Pose3D(x=x, y=y, z=z), marker_id=matched[0])
        logger.info(f"ArUco marker {matched[0]} confirmed at x={x:.2f} y={y:.2f} z={z:.2f}")




#  Lawnmower thread, searches, reads pos from UAVState
#  stops as soon as marker_confirmed is set, or pattern finishes

def run_lawnmower(state: UAVState, controller: StationaryLandingController, cfg):
    ''' wraps LawnmowerMission so it reads position from UAVState instead of holding a direct vo reference.'''

    # Adapter: gives LawnmowerMission a .pose() and .status interface
    # backed by UAVState, so lawnmower stays decoupled from VO internals.

    class StateVOAdapter:
        def __init__(self, state_accessor):
            self._state = state_accessor
        
        @property
        def status(self):
            (x, y, z, yaw), timestamp = self._state.get_vio_position()
            return "TRACKING" if timestamp > 0 else "INIT"
        
        def pose(self):
            (x, y, z, yaw), _ = self._state.get_vio_position()
            return np.array([x, y, z]), yaw
    
    vo_adapter = StateVOAdapter(state)
    
    try:
        # LawnmowerMissionRunner uses threading internally but runs in this process
        mission = LawnmowerMissionRunner(
            mav_master=controller.master,
            vo=vo_adapter,
            valid_ids=cfg.aruco.target_marker_id if isinstance(cfg.aruco.target_marker_id, list) else [cfg.aruco.target_marker_id]
        )
        
        logger.info("[LAWNMOWER] Starting search pattern")
        mission.run()
        
        # run() blocks until marker found or pattern complete
        if mission.search.valid_marker_confirmed:
            logger.info("[LAWNMOWER] Search complete — marker confirmed")
        else:
            logger.warning("[LAWNMOWER] Search complete — marker NOT found")
    
    finally:
        state.close()
        logger.info("[LAWNMOWER] Process exiting")


# =====
# Process 4: Landing (AprilTag precision)
# =====

def run_landing(lock, marker_confirmed, ugv_signal, hover_reached, cfg):
    """
    AprilTag precision landing process.
    
    Waits for marker_confirmed Event, then begins precision landing.
    Uses downward camera to track AprilTag on UGV.
    """
    logger.info("[LANDING] Process starting")
    
    state = UAVStateAccessor(lock, marker_confirmed, ugv_signal, hover_reached)
    
    logger.info("[LANDING] Waiting for ArUco marker confirmation...")
    marker_confirmed.wait()
    
    logger.info("[LANDING] Marker confirmed — starting AprilTag landing sequence")
    state.set_flight_mode(FlightMode.LAND)
    
    # Each process creates its own MAVLink connection
    controller = StationaryLandingController(
        cfg.pixhawk.connection_string,
        cfg.pixhawk.baud_rate
    )
    
    try:
        with dai.Device() as device:
            calibration = device.getCalibration()
            april_tag_detector = AprilTagDetector(calibration)
            
            with dai.Pipeline(device) as pipeline:
                cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
                rgb_out = cam_rgb.requestOutput(
                    size=(300, 300),
                    type=dai.ImgFrame.Type.NV12,
                    fps=30
                )
                q_rgb = rgb_out.createOutputQueue(maxSize=4, blocking=False)
                
                pipeline.start()
                
                last_tag_time = time.time()
                HOVER_TIMEOUT = 4.0
                SEARCH_TIMEOUT = 7.0
                
                logger.info("[LANDING] AprilTag tracking loop running")
                
                while pipeline.isRunning():
                    in_rgb = q_rgb.get()
                    if in_rgb is None:
                        continue
                    
                    frame = in_rgb.getCvFrame()
                    pose = april_tag_detector.get_tag_pose(frame)
                    
                    if pose is None:
                        time_lost = time.time() - last_tag_time
                        
                        if time_lost < HOVER_TIMEOUT:
                            controller.send_velocity(0, 0, 0)
                        elif time_lost < SEARCH_TIMEOUT:
                            logger.warning(f"[LANDING] Tag lost {time_lost:.1f}s, ascending")
                            controller.send_velocity(0, 0, -0.2)
                        else:
                            logger.error("[LANDING] Tag lost >7s, blind VIO descent")
                            controller.send_velocity(0, 0, 0.3)
                        
                        continue
                    
                    last_tag_time = time.time()
                    
                    cam_x, cam_y, cam_z = pose
                    body_x, body_y, body_z = controller.convert_camera_to_body_frame(
                        cam_x, cam_y, cam_z
                    )
                    
                    logger.info(f"[LANDING] body x={body_x:.2f} y={body_y:.2f} z={body_z:.2f}")
                    
                    if body_z < cfg.pixhawk.landing_threshold:
                        logger.info("[LANDING] Landing threshold reached — landing")
                        controller.stationary_landing()
                        time.sleep(5)
                        controller.disarm_motors()
                        break
                    
                    controller.adjust_velocity_and_send(body_x, body_y, body_z)
    
    finally:
        state.close()
        logger.info("[LANDING] Process exiting")


# =====
# Main Process
# =====

def main():
    parser = argparse.ArgumentParser(description="UAV autonomous mission — multiprocessing")
    parser.add_argument("--mode", choices=["scan", "land"], default="scan")
    args = parser.parse_args()
    
    cfg = load_config(mode=args.mode)
    
    logger.info(f"Starting UAV system in mode: {cfg.mode}")
    logger.info("Using multiprocessing architecture — 5 independent processes")
    
    # Create shared state (this process owns the shared memory)
    state_dict = create_shared_state()
    lock = state_dict["lock"]
    marker_confirmed = state_dict["marker_confirmed"]
    ugv_signal = state_dict["ugv_signal"]
    hover_reached = state_dict["hover_reached"]
    
    # Create state accessor for main process
    state = UAVStateAccessor(lock, marker_confirmed, ugv_signal, hover_reached)
    
    # Telemetry database path
    Path("flight_logs").mkdir(exist_ok=True)
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = f"flight_logs/flight_{log_timestamp}.db"
    
    # Main process handles arm/takeoff (before spawning children)
    logger.info("[MAIN] Connecting to Pixhawk for arm/takeoff")
    controller = StationaryLandingController(
        cfg.pixhawk.connection_string,
        cfg.pixhawk.baud_rate
    )
    
    logger.info("[MAIN] Arming and taking off")
    controller.change_flight_mode("GUIDED")
    controller.arm_motors()
    controller.takeoff_to_altitude(cfg.pixhawk.hover_altitude_m)
    
    state.set_flight_mode(FlightMode.SCAN)
    logger.info("[MAIN] Takeoff complete — spawning processes")
    
    # Spawn all processes
    processes = []
    
    # Process 1: SLAM
    p_slam = mp.Process(
        target=run_slam,
        args=(lock, marker_confirmed, ugv_signal, hover_reached, cfg),
        name="slam"
    )
    processes.append(p_slam)
    
    # Process 2: Vision
    p_vision = mp.Process(
        target=run_vision,
        args=(lock, marker_confirmed, ugv_signal, hover_reached, cfg),
        name="vision"
    )
    processes.append(p_vision)
    
    # Process 3: Lawnmower
    p_lawnmower = mp.Process(
        target=run_lawnmower,
        args=(lock, marker_confirmed, ugv_signal, hover_reached, cfg),
        name="lawnmower"
    )
    processes.append(p_lawnmower)
    
    # Process 4: Landing
    p_landing = mp.Process(
        target=run_landing,
        args=(lock, marker_confirmed, ugv_signal, hover_reached, cfg),
        name="landing"
    )
    processes.append(p_landing)
    
    # Process 5: Telemetry
    p_telemetry = mp.Process(
        target=telemetry_logger,
        args=(lock, db_path, cfg.pixhawk.connection_string, cfg.pixhawk.baud_rate),
        name="telemetry"
    )
    processes.append(p_telemetry)
    
    # Start all processes
    for p in processes:
        p.start()
        logger.info(f"[MAIN] Started process: {p.name}")
    
    logger.info(f"[MAIN] All processes running — telemetry logging to {db_path}")
    
    # Main monitoring loop
    try:
        while True:
            mode = state.get_flight_mode()
            (vio_x, vio_y, vio_z, vio_yaw), vio_ts = state.get_vio_position()
            (aruco_pos, marker_id) = state.get_aruco_pose()
            
            vio_str = f"{vio_x:.2f},{vio_y:.2f},{vio_z:.2f}" if vio_ts > 0 else "None"
            marker_str = f"ID:{marker_id}" if marker_id else "None"
            
            logger.info(
                f"[STATUS] mode={mode.name} | "
                f"vio={vio_str} | "
                f"marker={marker_str}"
            )
            
            # Check if lawnmower and landing are done
            if not p_lawnmower.is_alive() and not p_landing.is_alive():
                logger.info("[MAIN] Lawnmower and landing both finished — mission complete")
                break
            
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        logger.info("[MAIN] Keyboard interrupt — shutting down")
    
    finally:
        # Terminate all processes
        logger.info("[MAIN] Terminating all processes")
        for p in processes:
            if p.is_alive():
                p.terminate()
        
        # Wait for clean exit
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                logger.warning(f"[MAIN] Force killing {p.name}")
                p.kill()
        
        # Close state accessor
        state.close()
        
        # Clean up shared memory (only main process does this)
        cleanup_shared_state(state_dict)
        
        logger.info(f"[MAIN] Shutdown complete. Telemetry saved to {db_path}")
        logger.info("[MAIN] Done. Yabadabadoo!")


if __name__ == "__main__":

    # CRITICAL: must use 'spawn' method for multiprocessing on Raspberry Pi
    # 'fork' causes issues with DepthAI and OpenCV
    mp.set_start_method('spawn', force=True)
    main()

