''' Main file to run entire UAV '''

import time
import depthai as dai
import argparse
import threading
import numpy as np
import cv2

from vio_slam.vo_full_v3 import (
    VO_LK, LoopClosureORB, MavlinkVisionPublisher,
    RGB_SOCKET, LEFT_SOCKET, RIGHT_SOCKET, ENABLE_LOOP, KEYFRAME_INTERVAL,
    ENABLE_MAVLINK_VISION,FPS, W, H, IMU_HZ
)
# fps = 30.0, w, h = 640 x 400

from core.config import load_config
from core.state import UAVState, FlightMode, Pose3D
from core.log import get_logger
from core.json_utils import read_json

# Vision (from old repo uav-2526)
from vision.common.detectors.detector_manager import DetectorManager
from vision.common.video.camera_coordinate_transformer import CameraCoordinateTransformer
from vision.common.segmentation.uav_segmenter import FieldObstacleSegmenter, SegmenterConfig

# Landing (swap in test_move.py logic here later)
from landing.april_detector.april_tag_detector import AprilTagDetector
from landing.pixhawk_controller.stationary_landing_controller import StationaryLandingController
from landing.lawnmower import LawnmowerMission


logger = get_logger(__name__)


#  SLAM thread, wraps the VO pipeline from vo_full_v3.py.
#  Runs inside the shared dai.Device context.
def run_slam(pipeline: dai.Pipeline, device: dai.Device, state: UAVState, cfg):
    """Builds the stereo/IMU pipeline nodes and runs the VO loop.
    Publishes VIO position to state on every frame."""

    

    # Build stereo pipeline nodes (same as vo_full_v3.main(), lines 550-580~)
    cam_rgb = pipeline.create(dai.node.Camera).build(RGB_SOCKET)
    cam_left = pipeline.create(dai.node.Camera).build(LEFT_SOCKET)
    cam_right = pipeline.create(dai.node.Camera).build(RIGHT_SOCKET)

    stereo = pipeline.create(dai.node.StereoDepth)
    imu = pipeline.create(dai.node.IMU)
    sync= pipeline.create(dai.node.Sync)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setDepthAlign(
        dai.StereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT
    )

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
        def __init__(self, state: UAVState):
            self._state = state

        @property
        def status(self):
            vio = self._state.get_vio_position()
            return "TRACKING" if vio is not None else "INIT"

        def pose(self):
            vio = self._state.get_vio_position()
            if vio is None:
                return np.zeros(3), 0.0
            return np.array([vio.x, vio.y, vio.z]), vio.yaw

    vo_adapter = StateVOAdapter(state)

    mission = LawnmowerMission(
        vo=vo_adapter,
        mav_master=controller.master,
        # arm/takeoff already done in main() before threads starts
        auto_arm=False
    )

    logger.info("Lawnmower thread: starting search pattern")
    mission.start()
    while mission.state not in ("DONE", "LANDING"):
        if state.marker_confirmed.is_set():
            logger.info("ArUco confirmed, stopping lawnmower early")
            mission.stop()
            break
        time.sleep(0.5)

    logger.info(f"Lawnmower thread done (final state: {mission.state})")




#  Landing thread,
# TODO: update w/Aris new push, then use ugv_signal.wait() once UGV comms is ready.

def run_landing(pipeline: dai.Pipeline, device: dai.Device, state: UAVState, controller: StationaryLandingController, cfg):
    logger.info("Landing thread: waiting for ArUco confirmation...")
    state.marker_confirmed.wait()

    # logger.info("Waiting for UGV signal...")
    # state.ugv_signal.wait()

    logger.info("Starting april tag landing sequence")
    state.set_flight_mode(FlightMode.LAND)

    # Build a small dedicated pipeline output for downward-facing landing camera.
    # Using CAM_A at low res, april tag detection doesn't need high resolution.
    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgb_out = cam_rgb.requestOutput(
        size=(300, 300),
        type=dai.ImgFrame.Type.NV12,
        fps=30
    )
    q_rgb = rgb_out.createOutputQueue(maxSize=4, blocking=False)

    calibration = device.getCalibration()
    april_tag_detector = AprilTagDetector(calibration)

    last_tag_time  = time.time()
    HOVER_TIMEOUT  = 4.0
    SEARCH_TIMEOUT = 7.0

    logger.info("Landing loop running")
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
                logger.warning(f"Tag lost {time_lost:.1f}s,ascending to widen FOV")
                controller.send_velocity(0, 0, -0.2)

            else:
                logger.error("Tag lost >7s, blind VIO descent")
                controller.send_velocity(0, 0, 0.3)

            continue

        last_tag_time = time.time()

        cam_x, cam_y, cam_z = pose
        body_x, body_y, body_z = controller.convert_camera_to_body_frame(
            cam_x, cam_y, cam_z
        )

        logger.info(f"[LAND] body x={body_x:.2f} y={body_y:.2f} z={body_z:.2f}")

        if body_z < cfg.pixhawk.landing_threshold:
            logger.info("Landing threshold reached — landing")
            controller.stationary_landing()
            controller.disarm_motors()
            break

        controller.adjust_velocity_and_send(body_x, body_y, body_z)






def main():
    parser = argparse.ArgumentParser(description="UAV autonomous mission")
    parser.add_argument("--mode", choices=["scan", "land"], default="scan")
    args = parser.parse_args()
    cfg   = load_config(mode=args.mode)
    state = UAVState()

    logger.info(f"Starting in mode: {cfg.mode}")

    with dai.Device() as device:
        calibration = device.getCalibration()
        # Controller is shared between lawnmower + landing threads
        controller = StationaryLandingController(
            cfg.pixhawk.connection_string,
            cfg.pixhawk.baud_rate
        )

        with dai.Pipeline(device) as pipeline:
            # Arm + takeoff before any threads start so the drone is
            # airborne before the lawnmower and vision threads kick in.
            controller.change_flight_mode("GUIDED")
            controller.arm_motors()
            controller.takeoff_to_altitude(cfg.pixhawk.hover_altitude_m)

            state.set_flight_mode(FlightMode.SCAN)

            # SLAM keeps running, feeds position to UAVState
            threading.Thread(
                target=run_slam,
                args=(pipeline, device, state, cfg),
                daemon=True, name="slam"
            ).start()

            # Vision scans for ArUco, sets marker_confirmed when found
            threading.Thread(
                target=run_vision,
                args=(pipeline, state, cfg),
                daemon=True, name="vision"
            ).start()

            # Lawnmower, flies search pattern, stops when marker confirmed
            threading.Thread(
                target=run_lawnmower,
                args=(state, controller, cfg),
                daemon=True, name="lawnmower"
            ).start()

            # Landing, waits for marker_confirmed, then april tag land
            threading.Thread(
                target=run_landing,
                args=(pipeline, device, state, controller, cfg),
                daemon=True, name="landing"
            ).start()

            pipeline.start()

            try:
                while pipeline.isRunning():
                    mode      = state.get_flight_mode()
                    vio       = state.get_vio_position()
                    pose, mid = state.get_aruco_pose()

                    logger.info(
                        f"[STATUS] mode={mode.name} | "
                        f"vio={f'{vio.x:.2f},{vio.y:.2f},{vio.z:.2f}' if vio else 'None'} | "
                        f"marker={'ID:'+str(mid) if mid is not None else 'None'}"
                    )
                    time.sleep(1.0)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt — shutting down")

    logger.info("Done. Yabadabadoo!")


if __name__ == "__main__":
    main()