# main.py
import argparse
import threading
import time
import numpy as np
import cv2
import depthai as dai

from core.config import load_config
from core.state import UAVState, FlightMode
from core.logger import get_logger

# Vision (from uav-2526 / auto_uav.py)
from vision.common.detectors.detector_manager import DetectorManager
from vision.common.video.camera_coordinate_transformer import CameraCoordinateTransformer
from vision.common.segmentation.uav_segmenter import FieldObstacleSegmenter, SegmenterConfig
from vision.common.utils.json_utils import read_json
import cv2
from vision.common.detectors.detector_manager import DetectorManager
from vision.common.video.camera_coordinate_transformer import CameraCoordinateTransformer
from vision.common.segmentation.uav_segmenter import FieldObstacleSegmenter, SegmenterConfig
from core.state import Pose3D

# SLAM (from vo_full_v3.py — broken out of its main())
from slam.vo_full_v3 import VO_LK, LoopClosureORB, MavlinkVisionPublisher
from slam.vo_full_v3 import (
        VO_LK, LoopClosureORB, MavlinkVisionPublisher,
        ENABLE_LOOP, KEYFRAME_INTERVAL, ENABLE_MAVLINK_VISION,
        FPS, W, H, IMU_HZ
    )

# Landing (stub — swap in test_move.py logic here later)
# from landing.pixhawk.controller import PixhawkController

logger = get_logger(__name__)


# ─────────────────────────────────────────────
#  SLAM thread
#  Wraps the VO pipeline from vo_full_v3.py.
#  Runs inside the shared dai.Device context.
# ─────────────────────────────────────────────
def run_slam(pipeline: dai.Pipeline, device: dai.Device, state: UAVState, cfg):
    """Builds the stereo/IMU pipeline nodes and runs the VO loop.
    Publishes VIO position to state on every frame."""

    

    # Build stereo pipeline nodes (same as vo_full_v3.main())
    cam_rgb   = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_left  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

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
        calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, W, H),
        dtype=np.float64
    )

    vo   = VO_LK(K)
    loop = LoopClosureORB() if ENABLE_LOOP else None

    sync_q = sync.out.createOutputQueue()
    imu_q  = imu.out.createOutputQueue(maxSize=50, blocking=False)

    frame_id = 0
    t0 = time.time()

    logger.info("SLAM thread running")

    while pipeline.isRunning():
        # IMU update
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

        # Publish to shared state
        from core.state import Pose3D
        state.set_vio_position(Pose3D(
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
            yaw=yaw_vis
        ))

        # Loop closure
        if ENABLE_LOOP and loop and vo.status == "TRACKING":
            if (frame_id % KEYFRAME_INTERVAL) == 0:
                rgb = msg_group["rgb"].getCvFrame()
                loop.add_keyframe(rgb, pos, frame_id, time.time() - t0)
            info = loop.check_loop(msg_group["rgb"].getCvFrame(), pos, frame_id)
            if info:
                vo.apply_soft_correction(info["matched_pose"])

        frame_id += 1


# ─────────────────────────────────────────────
#  Vision thread
#  Wraps AutoUav logic from auto_uav.py.
#  Shares the same dai.Device / pipeline.
# ─────────────────────────────────────────────
def run_vision(pipeline: dai.Pipeline, state: UAVState, cfg):
    """Runs ArUco detection on the RGB stream.
    Sets state.marker_confirmed when the target marker is found."""

    # RGB node — request a separate lower-res output for detection
    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgb_out = cam_rgb.requestOutput(size=(640, 480), fps=cfg.camera.color_fps)
    q_rgb   = rgb_out.createOutputQueue(maxSize=4, blocking=False)

    detector   = DetectorManager(cfg.detector).get_detector()
    transformer = CameraCoordinateTransformer(cfg.video)
    segmenter  = FieldObstacleSegmenter(SegmenterConfig(process_interval_sec=0.5))

    target_ids = cfg.aruco.target_marker_id  # e.g. [3, 7]

    logger.info("Vision thread running")

    while pipeline.isRunning():
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

        state.set_aruco_pose(
            Pose3D(x=x, y=y, z=z),
            marker_id=matched[0]
        )
        logger.info(f"ArUco marker {matched[0]} confirmed at x={x:.2f} y={y:.2f} z={z:.2f}")


# ─────────────────────────────────────────────
#  Landing thread (stub)
#  Will be replaced with test_move.py logic.
# ─────────────────────────────────────────────
def run_landing(state: UAVState, cfg):
    """Waits for marker confirmation + UGV signal, then executes landing.
    TODO: replace stub with test_move.py logic."""

    logger.info("Landing thread: waiting for marker confirmation...")
    state.marker_confirmed.wait()   # blocks until vision fires this
    logger.info("Marker confirmed. Waiting for UGV signal...")

    state.ugv_signal.wait()         # blocks until comms fires this
    logger.info("UGV signal received. Beginning landing sequence...")

    state.set_flight_mode(FlightMode.LAND)

    # TODO: plug in test_move.py / PixhawkController here
    # controller = PixhawkController(cfg.pixhawk.connection_string, cfg.pixhawk.baud_rate)
    # controller.land_on_tag(state)


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="UAV autonomous mission")
    parser.add_argument("--mode", choices=["scan", "land"], default="scan")
    args = parser.parse_args()

    cfg   = load_config(mode=args.mode)
    state = UAVState()

    logger.info(f"Starting UAV in mode: {cfg.mode}")
    state.set_flight_mode(FlightMode.SCAN)

    # One device context shared by all subsystems
    with dai.Device() as device:
        with dai.Pipeline(device) as pipeline:

            # Launch SLAM in background thread
            slam_thread = threading.Thread(
                target=run_slam,
                args=(pipeline, device, state, cfg),
                daemon=True,
                name="slam"
            )
            slam_thread.start()

            # Launch Vision in background thread
            vision_thread = threading.Thread(
                target=run_vision,
                args=(pipeline, state, cfg),
                daemon=True,
                name="vision"
            )
            vision_thread.start()

            # Launch Landing in background thread
            landing_thread = threading.Thread(
                target=run_landing,
                args=(state, cfg),
                daemon=True,
                name="landing"
            )
            landing_thread.start()

            # Start pipeline — blocks until quit
            pipeline.start()

            try:
                while pipeline.isRunning():
                    mode = state.get_flight_mode()
                    vio  = state.get_vio_position()
                    pose, mid = state.get_aruco_pose()

                    logger.debug(
                        f"[STATUS] mode={mode.name} | "
                        f"vio={f'{vio.x:.2f},{vio.y:.2f},{vio.z:.2f}' if vio else 'None'} | "
                        f"marker={'ID:'+str(mid) if mid is not None else 'None'}"
                    )
                    time.sleep(1.0)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt — shutting down")

    logger.info("Done.")


if __name__ == "__main__":
    main()