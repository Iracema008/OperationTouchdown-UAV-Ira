''' Bridge between vo_full_v3.py & OAK-D S2 camera into ArduPilot SITL'''


# PYTHONPATH=. python3 sitl/sitl_vo_bridge.py
# QGroundControl connects w/UDP 14550

import math
import os
import sys
import depthai as dai
import time
import threading
import importlib

import cv2
import numpy as np
from pymavlink import mavutil
from vision.common.detectors.opencv_helpers import Cv2Detector


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# same ids from auto_uav.py
TARGET_IDS = [3, 7]
SAVED_FOLDER = "saved-aruco"

# SITL connection
SITL_UDP_HOST = "127.0.0.1"
SITL_UDP_PORT = 14551
VISION_RATE_HZ = 30.0

def _time_usec(boot_wall: float) -> int:
    return int((time.time() - boot_wall) * 1e6)


class ArucoHandler:
    """ Wraps Cv2Detector with the check_ids / datapack_save"""
    def __init__(self, target_ids: list, saved_folder: str):
        self.target_ids = target_ids
        self.saved_folder = saved_folder
        self.detector = Cv2Detector()
        self.correct_marker = False
        self.marker_detected_before = False

        os.makedirs(saved_folder, exist_ok=True)
        print(f"[ArUco] Detector ready. Target IDs: {target_ids}")
        print(f"[ArUco] Saves go to: {os.path.abspath(saved_folder)}/")

    def process_frame(self, frame: np.ndarray) -> tuple:
        corners, ids, _ = self.detector.detect(frame, print_corners=False)
        flat_ids = None
        if ids is not None:
            flat_ids = ids.flatten().tolist()

        self.check_ids(frame, flat_ids)

        annotated = self.detector.draw_detections(frame, corners, ids)
        return corners, ids, annotated

    # the same, 
    def check_ids(self, frame: np.ndarray, found_ids):
        if found_ids is None:
            return

        self.correct_marker = any(id_ in self.target_ids for id_ in found_ids)

        if self.correct_marker and not self.marker_detected_before:
            print(f"[ArUco] Correct marker FOUND: {found_ids}")
            self.datapack_save(frame, found_ids)
            self.marker_detected_before = True

        return found_ids

    # right now just keeps overwritting to avoid big logs
    def datapack_save(self, frame: np.ndarray, flight_data):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not os.path.exists(self.saved_folder):
            print(f"[ArUco] Folder missing — creating {self.saved_folder}")
            os.makedirs(self.saved_folder, exist_ok=True)

        file_save = os.path.join(self.saved_folder, "pack_image.png")
        cv2.imwrite(file_save, gray)
        print(f"[ArUco] Saved image to: {file_save}")

    def reset_detection(self):
       # cll to save again
        self.marker_detected_before = False
        self.correct_marker = False

class SITLVOBridge:
    ''' Reads pose from running VO_LK instance, treams VISION_POSITION_ESTIMATE to ArduPilot SITL, using UDP'''
    def __init__(self, vo):
        self.vo = vo
        self._stop = threading.Event()
        self._thread = None
        self._period = 1.0 / VISION_RATE_HZ
        self._sent = 0
        self._aligned = False
        self._pix_yaw0 = None
        self._yaw_offset = 0.0

        print(f"[SITLBridge] Connecting → udp:{SITL_UDP_HOST}:{SITL_UDP_PORT} ...")
        self.master = mavutil.mavlink_connection(
            f"udpout:{SITL_UDP_HOST}:{SITL_UDP_PORT}",
            source_system=255,
        )
        print("[SITLBridge] Waiting for SITL heartbeat ...")
        self.master.wait_heartbeat(timeout=15)
        self.boot_wall = time.time()
        print(f"[SITLBridge] Heartbeat OK — streaming at {VISION_RATE_HZ} Hz.")

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="SITLVOBridge"
        )
        self._thread.start()
        print("[SITLBridge] Vision stream thread started.")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        print(f"[SITLBridge] Stopped. Total messages sent: {self._sent}")



    def _run(self):
        last_send = 0.0
        while not self._stop.is_set():
            now = time.time()
            if now - last_send < self._period:
                time.sleep(0.005)
                continue
            last_send = now

            # Poll attitude for yaw alignment
            att = self.master.recv_match(type="ATTITUDE", blocking=False)
            if att is not None and self._pix_yaw0 is None:
                self._pix_yaw0 = float(att.yaw)

            if self.vo.status != "TRACKING":
                continue

            pos, yaw_vis_deg = self.vo.pose()

            # One-time yaw alignment (mirrors MavlinkVisionPublisher in vo_full_v3.py)
            if not self._aligned and self._pix_yaw0 is not None:
                vo_yaw0 = math.radians(float(yaw_vis_deg))
                self._yaw_offset = self._pix_yaw0 - vo_yaw0
                self._aligned = True
                print(
                    f"[SITLBridge] Yaw aligned. "
                    f"offset={math.degrees(self._yaw_offset):+.2f}°"
                )

            # VO frame to  NED (same mapping as vo_full_v3.py vo_pose_to_ned)
            # when positioning on qgc long is negitve 
            # x_vo=right, y_vo=down, z_vo=forward → N=z_vo, E=x_vo, D=y_vo
            x_vo, y_vo, z_vo = float(pos[0]), float(pos[1]), float(pos[2])
            ned = np.array([z_vo, x_vo, y_vo], dtype=np.float64)

            if self._aligned:
                c = math.cos(self._yaw_offset)
                s = math.sin(self._yaw_offset)
                ne = np.array([
                    c * ned[0] - s * ned[1],
                    s * ned[0] + c * ned[1],
                ])
                ned[0], ned[1] = float(ne[0]), float(ne[1])

            yaw_ned = (
                self._pix_yaw0
                if self._pix_yaw0 is not None
                else math.radians(float(yaw_vis_deg))
            )

            self.master.mav.vision_position_estimate_send(
                #n, e,d, rool, pitch, yaw
                _time_usec(self.boot_wall), float(ned[0]), float(ned[1]),
                float(ned[2]), 0.0, 0.0, float(yaw_ned),
            )
            self._sent += 1

            # Logs every 5 seconds
            if self._sent % int(VISION_RATE_HZ * 5) == 0:
                print(
                    f"[SITLBridge] N={ned[0]:+.2f} E={ned[1]:+.2f} D={ned[2]:+.2f} "
                    f"yaw={math.degrees(yaw_ned):+.1f}° | msgs={self._sent}"
                )


def main():
    try:
        vo_mod = importlib.import_module("vio_slam.vo_full_v3")
    except ModuleNotFoundError:
        try:
            vo_mod = importlib.import_module("vo_full_v3")
        except ModuleNotFoundError:
            print(
                "[SITLBridge] ERROR: Cannot find vo_full_v3.\n"
                "Run from UAV-ira/ root:\n"
                "  PYTHONPATH=. python3 sitl/sitl_vo_bridge.py"
            )
            sys.exit(1)

    FPS = vo_mod.FPS
    W, H = vo_mod.W, vo_mod.H
    RGB_SOCKET = vo_mod.RGB_SOCKET
    LEFT_SOCKET = vo_mod.LEFT_SOCKET
    RIGHT_SOCKET=  vo_mod.RIGHT_SOCKET

    
    aruco = ArucoHandler(target_ids=TARGET_IDS, saved_folder=SAVED_FOLDER)
    bridge = None

    with dai.Device() as device:
        with dai.Pipeline(device) as pipeline:
            cam_rgb   = pipeline.create(dai.node.Camera).build(RGB_SOCKET)
            cam_left  = pipeline.create(dai.node.Camera).build(LEFT_SOCKET)
            cam_right = pipeline.create(dai.node.Camera).build(RIGHT_SOCKET)

            stereo = pipeline.create(dai.node.StereoDepth)
            imu    = pipeline.create(dai.node.IMU)
            sync   = pipeline.create(dai.node.Sync)

            stereo.setDefaultProfilePreset(
                dai.node.StereoDepth.PresetMode.FAST_DENSITY
            )
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(True)
            stereo.setDepthAlign(
                dai.StereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT
            )

            #rn just uses camera imu, maybe correct w pixhawwks? 
            imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, vo_mod.IMU_HZ)
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

            # Camera calibration
            calib = device.getCalibration()
            K = np.array(
                calib.getCameraIntrinsics(LEFT_SOCKET, W, H), dtype=np.float64
            )
            print(f"[SITLBridge] Camera intrinsics loaded:\n{K}\n")

            # VO W/optional loop closure
            vo = vo_mod.VO_LK(K)
            loop_closure = vo_mod.LoopClosureORB() if vo_mod.ENABLE_LOOP else None

            # Start the MAVLink bridge thread
            bridge = SITLVOBridge(vo)
            bridge.start()

            sync_q = sync.out.createOutputQueue()
            imu_q  = imu.out.createOutputQueue(maxSize=50, blocking=False)
            pipeline.start()
            t0       = time.time()
            frame_id = 0

            print("\n[SITLBridge] Running — press Q in the camera window to quit.\n")
            while pipeline.isRunning():
                t_sec = time.time() - t0
                # imu
                try:
                    imu_msgs = imu_q.tryGetAll()
                except Exception:
                    imu_msgs = []
                for msg in imu_msgs:
                    for pkt in msg.packets:
                        vo.update_imu(pkt.gyroscope.z)

                # Synced frames
                msg_group = sync_q.get()
                if msg_group is None:
                    continue

                rgb   = msg_group["rgb"].getCvFrame()
                gray  = msg_group["left"].getCvFrame()
                depth = msg_group["depth"].getFrame()

                if rgb is None or gray is None or depth is None:
                    continue

                # VO update
                vo.process(gray, depth)

                # Loop closure
                if loop_closure and vo.status == "TRACKING":
                    if frame_id % vo_mod.KEYFRAME_INTERVAL == 0:
                        pos_lc, _ = vo.pose()
                        loop_closure.add_keyframe(rgb, pos_lc, frame_id, t_sec)
                    info = loop_closure.check_loop(rgb, vo.pose()[0], frame_id)
                    if info:
                        vo.apply_soft_correction(info["matched_pose"])

                pos, yaw_vis = vo.pose()
                corners, ids, annotated_frame = aruco.process_frame(rgb)
                #visual of camera
                vis = annotated_frame if annotated_frame is not None else rgb.copy()
                cv2.putText(
                    vis, f"SITL VO BRIDGE | VO: {vo.status}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                )
                cv2.putText(
                    vis,
                    f"N={pos[2]:+.2f}m  E={pos[0]:+.2f}m  D={pos[1]:+.2f}m",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2,
                )
                cv2.putText(
                    vis,
                    f"tracked={vo.num_tracked}  inliers={vo.inliers}  "
                    f"msgs→SITL={bridge._sent}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1,
                )

                aruco_color = (0, 220, 60) if aruco.correct_marker else (0, 80, 220)
                aruco_txt = (
                    "ArUco: VALID marker saved!"
                    if aruco.marker_detected_before
                    else f"ArUco: watching for IDs {TARGET_IDS}"
                )
                cv2.putText(
                    vis, aruco_txt,
                    (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.48, aruco_color, 1,
                )

                align_txt = "yaw aligned" if bridge._aligned else "aligning yaw..."
                cv2.putText(
                    vis, f"SITL: {align_txt}",
                    (10, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1,
                )

                cv2.imshow("SITL VO Bridge + ArUco", vis)
                frame_id += 1

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()
    if bridge:
        bridge.stop()
    print("[SITLBridge] Done.")


if __name__ == "__main__":
    main()