''' Main file to run entire UAV, now w/multiprocessing '''

# python main.py --mode scan

import time
import argparse
import multiprocessing as mp
from multiprocessing import Array
from datetime import datetime
from pathlib import Path

import numpy as np
import depthai as dai
import cv2 as _cv2

# Multiprocessing safe state
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

from vision.common.detectors.detector_manager import DetectorManager
from vision.common.video.camera_coordinate_transformer import CameraCoordinateTransformer
from mission.pixhawk_controller.stationary_landing_controller import StationaryLandingController
# from vision.common.detectors.april_detector.april_tag_detector import AprilTagDetector
from mission.lawnmower import run_lawnmower_mission
from telemetry.telemetry_logger import telemetry_logger

logger = get_logger(__name__)

# Number of consecutive valid ID detections before SA replan is triggered.
# Must be less than ArucoConfig.min_consecutive_detections (full confirm = 3)
UNCERTAIN_DETECTION_THRESHOLD = 2


def run_slam(lock, marker_confirmed, ugv_signal, hover_reached, cfg):
    """
    Visual-inertial odometry process.
    Runs stereo depth + optical flow + loop closure.
    Writes position to shared memory every frame.
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


def save_marker_snapshot(frame, corners, ids, marker_id: int,
                          log_timestamp: str):
    """Save annotated color frame to flight_logs/markers/ on confirmation."""
    save_dir = Path("flight_logs/markers")
    save_dir.mkdir(parents=True, exist_ok=True)

    annotated = frame.copy()
    annotated = _cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

    filename = save_dir / f"marker_{marker_id}_{log_timestamp}.png"
    _cv2.imwrite(str(filename), annotated)
    logger.info(f"[VISION] Marker snapshot saved → {filename}")


def run_vision(lock, marker_confirmed, ugv_signal, hover_reached, cfg,
               log_timestamp: str, uncertain_pos):
    """
    ArUco marker detection process.

    Two signal levels:
        1. UNCERTAIN (2 consecutive detections of valid ID):
               Sets uncertain_pos[0]=north, uncertain_pos[1]=east,
               uncertain_pos[2]=1.0 → triggers SA replan in lawnmower
               Saves snapshot at this point too for reference

        2. CONFIRMED (min_consecutive_detections met):
               Saves annotated snapshot
               Writes pose to shared memory
               Sets marker_confirmed Event → triggers landing

    uncertain_pos layout: multiprocessing.Array('d', [north, east, flag])
        north, east : VO position at time of uncertain detection
        flag        : 0.0 = nothing, 1.0 = uncertain detection fired
    """
    logger.info("[VISION] Process starting")

    state = UAVStateAccessor(lock, marker_confirmed, ugv_signal, hover_reached)

    target_ids = cfg.aruco.target_marker_id
    if not isinstance(target_ids, list):
        target_ids = [target_ids]

    confirm_threshold  = cfg.aruco.min_consecutive_detections  # 3
    uncertain_threshold = UNCERTAIN_DETECTION_THRESHOLD          # 2

    detector    = DetectorManager(cfg.detector).get_detector()
    transformer = CameraCoordinateTransformer(cfg.video)

    # Per-ID consecutive detection counters
    consec_counts = {}          # {marker_id: int}
    uncertain_fired = False     # only fire uncertain signal once

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
                        # Reset all consecutive counters on no detection
                        consec_counts.clear()
                        continue

                    flat_ids = ids.flatten().tolist()
                    matched  = [i for i in flat_ids if i in target_ids]

                    # Reset counters for IDs not seen this frame
                    for mid in list(consec_counts):
                        if mid not in flat_ids:
                            consec_counts[mid] = 0

                    if not matched:
                        continue

                    # Update consecutive counter for each matched valid ID
                    for mid in matched:
                        consec_counts[mid] = consec_counts.get(mid, 0) + 1
                        count = consec_counts[mid]

                        logger.debug(
                            f"[VISION] ID={mid} consecutive={count}/"
                            f"{confirm_threshold}"
                        )

                        if (count == uncertain_threshold
                                and not uncertain_fired
                                and not marker_confirmed.is_set()):

                            # Compute detection position from camera
                            center_x = (
                                corners[0][0][0][0] +
                                (corners[0][0][2][0] - corners[0][0][0][0]) / 2
                            )
                            center_y = (
                                corners[0][0][0][1] +
                                (corners[0][0][2][1] - corners[0][0][0][1]) / 2
                            )
                            x, y, z = transformer.transform(
                                (center_x, center_y), 10
                            )

                            # Write to uncertain_pos for lawnmower to read
                            uncertain_pos[0] = x    # north approx
                            uncertain_pos[1] = y    # east approx
                            uncertain_pos[2] = 1.0  # flag: replan now

                            uncertain_fired = True
                            logger.info(
                                f"[VISION] Uncertain detection ID={mid} "
                                f"({count} consecutive) — "
                                f"SA replan signal sent "
                                f"x={x:.2f} y={y:.2f}"
                            )

                            # Save snapshot at uncertain detection too
                            save_marker_snapshot(
                                frame, corners, ids, mid,
                                f"{log_timestamp}_uncertain"
                            )

                        if count >= confirm_threshold:
                            center_x = (
                                corners[0][0][0][0] +
                                (corners[0][0][2][0] - corners[0][0][0][0]) / 2
                            )
                            center_y = (
                                corners[0][0][0][1] +
                                (corners[0][0][2][1] - corners[0][0][0][1]) / 2
                            )
                            x, y, z = transformer.transform(
                                (center_x, center_y), 10
                            )

                            save_marker_snapshot(
                                frame, corners, ids, mid, log_timestamp
                            )
                            state.set_aruco_pose(x, y, z, mid)
                            logger.info(
                                f"[VISION] ArUco ID={mid} CONFIRMED "
                                f"({count} consecutive) — "
                                f"x={x:.2f} y={y:.2f} z={z:.2f}"
                            )

    finally:
        state.close()
        logger.info("[VISION] Process exiting")


def run_lawnmower(lock, marker_confirmed, ugv_signal, hover_reached, cfg,
                  uncertain_pos, planner="grid"):
    """
    Lawnmower search with SA path optimisation.

    Pre-flight: SA optimises the waypoint visitation order.
    Mid-flight: SA replans remaining waypoints if uncertain_pos flag fires
                (set by run_vision on 2 consecutive valid ID detections).
    On confirm: flies to marker VO position, descends, lands.

    Sole MAVLink connection during testing — run_landing is disabled.

    TODO: When re-enabling AprilTag landing:
        - Comment out stationary_landing()/disarm_motors() in lawnmower.py
        - Uncomment p_landing in main()
        - Move controller to run_landing only
    """
    logger.info("[LAWNMOWER] Process starting")
    state = UAVStateAccessor(lock, marker_confirmed, ugv_signal, hover_reached)

    # Sole MAVLink connection — arm, takeoff, sweep, and land all happen here.
    # This avoids UDP port conflicts in SITL where two processes can't bind
    # the same port. On real hardware (serial), this also works fine.
    controller = StationaryLandingController(
        cfg.pixhawk.connection_string,
        cfg.pixhawk.baud_rate
    )

    logger.info("[LAWNMOWER] Arming and taking off")
    controller.change_flight_mode("GUIDED")
    controller.arm_motors()
    controller.takeoff_to_altitude(cfg.pixhawk.hover_altitude_m)

    state.set_flight_mode(FlightMode.SCAN)
    logger.info("[LAWNMOWER] Takeoff complete — starting search")

    # -----------------------------------------------------------------
    # VO ADAPTER — swap these two blocks depending on test mode:
    #
    # SITL testing (no camera): SITLVOAdapter reads position directly
    #   from Pixhawk LOCAL_POSITION_NED telemetry
    #
    # Real flight (OAK-D connected): StateVOAdapter reads VIO position
    #   from shared memory written by the SLAM process
    # -----------------------------------------------------------------

    class SITLVOAdapter:
        """
        Reads position from Pixhawk telemetry (LOCAL_POSITION_NED).
        Use during SITL testing when no OAK-D camera is available.
        """
        def __init__(self, master):
            self._master = master

        def pose(self):
            msg = self._master.recv_match(
                type='LOCAL_POSITION_NED',
                blocking=True,
                timeout=1
            )
            if msg:
                return np.array([msg.y, 0.0, msg.x]), 0.0
            return np.array([0.0, 0.0, 0.0]), 0.0

    class StateVOAdapter:
        """
        Reads VIO position from shared memory (written by SLAM process).
        Use during real flight when OAK-D is connected and SLAM is running.
        """
        def __init__(self, state_accessor):
            self._state = state_accessor

        @property
        def status(self):
            (x, y, z, yaw), timestamp = self._state.get_vio_position()
            return "TRACKING" if timestamp > 0 else "INIT"

        def pose(self):
            (x, y, z, yaw), _ = self._state.get_vio_position()
            return np.array([x, y, z]), yaw

    # SITL: comment out for real flight
   #vo_adapter = SITLVOAdapter(controller.master)

    # REAL FLIGHT: uncomment below and comment out SITLVOAdapter line above
    vo_adapter = StateVOAdapter(state)

    def on_marker_confirmed(north, east):
        logger.info(
            f"[LAWNMOWER] Landing at VO N={north:.2f} E={east:.2f}"
        )

    try:
        valid_ids = (
            cfg.aruco.target_marker_id
            if isinstance(cfg.aruco.target_marker_id, list)
            else [cfg.aruco.target_marker_id]
        )

        mission_state = run_lawnmower_mission(
            mav_master=controller.master,
            vo=vo_adapter,
            valid_ids=valid_ids,
            marker_confirmed=marker_confirmed,
            uncertain_pos=uncertain_pos,
            controller=controller,
            planner=planner,
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
# Process 4: AprilTag Precision Landing — DISABLED during lawnmower testing
# ===========================================================================
'''
def run_landing(lock, marker_confirmed, ugv_signal, hover_reached, cfg):
    ...
    # Uncomment and implement when lawnmower is validated.
    # See previous version of this file for full implementation.
'''


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="UAV autonomous mission — multiprocessing"
    )
    parser.add_argument("--mode", choices=["scan", "land"], default="scan")
    parser.add_argument(
        "--planner", choices=["grid", "sa"], default="grid",
        help="Path planner: 'grid' = plain lawnmower, 'sa' = simulated annealing"
    )
    args = parser.parse_args()

    cfg = load_config(mode=args.mode)

    logger.info(f"Starting UAV system in mode: {cfg.mode}")
    logger.info(f"Path planner: {args.planner}")
    logger.info("Using multiprocessing architecture")

    state_dict       = create_shared_state()
    lock             = state_dict["lock"]
    marker_confirmed = state_dict["marker_confirmed"]
    ugv_signal       = state_dict["ugv_signal"]
    hover_reached    = state_dict["hover_reached"]

    state = UAVStateAccessor(lock, marker_confirmed, ugv_signal, hover_reached)

    # ------------------------------------------------------------------
    # uncertain_pos — shared between run_vision and run_lawnmower
    # Layout: Array('d', [north, east, flag])
    #   north, east : VO position at uncertain detection
    #   flag        : 0.0 = not fired, 1.0 = SA replan requested
    # ------------------------------------------------------------------
    uncertain_pos = Array('d', [0.0, 0.0, 0.0])

    Path("flight_logs").mkdir(exist_ok=True)
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = f"flight_logs/flight_{log_timestamp}.db"

    logger.info("[MAIN] Spawning processes (arm/takeoff handled by lawnmower)")

    processes = []

    # Process 1: SLAM
    
    p_slam = mp.Process(
        target=run_slam,
        args=(lock, marker_confirmed, ugv_signal, hover_reached, cfg),
        name="slam"
    )
    processes.append(p_slam)
    

    
    # Process 2: Vision
    # Passes uncertain_pos — sets flag on 2 consecutive valid detections
    
    p_vision = mp.Process(
        target=run_vision,
        args=(lock, marker_confirmed, ugv_signal, hover_reached, cfg,
              log_timestamp, uncertain_pos),
        name="vision"
    )
    processes.append(p_vision)
    
    
    # Process 3: Lawnmower
    # Receives uncertain_pos — triggers SA replan when flag fires
    p_lawnmower = mp.Process(
        target=run_lawnmower,
        args=(lock, marker_confirmed, ugv_signal, hover_reached, cfg,
              uncertain_pos, args.planner),
        name="lawnmower"
    )
    processes.append(p_lawnmower)

    # Process 4: Landing — DISABLED during lawnmower testing
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

    try:
        while True:
            mode = state.get_flight_mode()
            (vio_x, vio_y, vio_z, vio_yaw), vio_ts = state.get_vio_position()
            (aruco_pos, marker_id) = state.get_aruco_pose()

            vio_str    = f"{vio_x:.2f},{vio_y:.2f},{vio_z:.2f}" if vio_ts > 0 else "None"
            marker_str = f"ID:{marker_id}" if marker_id else "None"
            sa_str     = f"SA-replan fired at N={uncertain_pos[0]:.2f} E={uncertain_pos[1]:.2f}" \
                         if uncertain_pos[2] == 1.0 else "SA-replan pending"

            logger.info(
                f"[STATUS] mode={mode.name} | "
                f"vio={vio_str} | "
                f"marker={marker_str} | "
                f"{sa_str}"
            )

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
    mp.set_start_method('spawn', force=True)
    main()