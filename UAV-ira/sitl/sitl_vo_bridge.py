"""
sitl_vo_bridge.py

Bridges your REAL vo_full_v3.py + OAK-D camera into ArduPilot SITL.

Instead of talking to a serial port, this connects to SITL over UDP
and sends VISION_POSITION_ESTIMATE messages so ArduPilot believes
it has a real VO system — without flying.

Architecture:
    OAK-D camera
        └── vo_full_v3.py  (VO_LK instance)
                └── sitl_vo_bridge.py  (this file — reads vo.pose(), sends to SITL)
                        └── ArduPilot SITL  (udp://127.0.0.1:14551)
                                └── QGroundControl  (udp://127.0.0.1:14550)

How to run:
    Terminal 1:  sim_vehicle.py -v ArduCopter --console --map
    Terminal 2:  python3 sitl_vo_bridge.py
    (vo_full_v3.py is imported and run as a thread inside this script)

SITL connection:
    MAVProxy default — SITL listens on udp:127.0.0.1:14551 for secondary GCS/scripts.
    If that port is taken, change SITL_UDP_PORT below.
"""

import time
import math
import threading
import numpy as np
from pymavlink import mavutil

# ─── CONFIG ──────────────────────────────────────────────────────────────────

SITL_UDP_PORT   = 14551          # MAVProxy forwards here by default
SITL_UDP_HOST   = "127.0.0.1"
VISION_RATE_HZ  = 30.0           # How often to send VISION_POSITION_ESTIMATE

# Set True once you've set the EK3 params (Part 1 of the guide).
# Set False to skip arming checks and just watch the position stream.
AUTO_ARM_TEST   = False

# ArUco — set True to enable marker detection alongside VO
ENABLE_ARUCO    = True
ARUCO_VALID_IDS = [3, 7, 12]     # ← edit this list

# ─────────────────────────────────────────────────────────────────────────────


def _time_usec(boot_wall: float) -> int:
    return int((time.time() - boot_wall) * 1e6)


def _rot_to_yaw(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])


class SITLVOBridge:
    """
    Reads pose from a running VO_LK instance and streams
    VISION_POSITION_ESTIMATE to ArduPilot SITL over UDP.
    """

    def __init__(self, vo, aruco_detector=None):
        self.vo = vo
        self.aruco_detector = aruco_detector
        self._stop = threading.Event()
        self._thread = None
        self._period = 1.0 / VISION_RATE_HZ

        print(f"[SITLBridge] Connecting to SITL at udp:{SITL_UDP_HOST}:{SITL_UDP_PORT} ...")
        self.master = mavutil.mavlink_connection(
            f"udpout:{SITL_UDP_HOST}:{SITL_UDP_PORT}",
            source_system=255,
        )
        print("[SITLBridge] Waiting for SITL heartbeat ...")
        self.master.wait_heartbeat(timeout=15)
        self.boot_wall = time.time()
        print(f"[SITLBridge] Heartbeat OK — SITL is alive. Starting vision stream at {VISION_RATE_HZ} Hz.")

        self._sent = 0
        self._yaw_offset = 0.0   # set after alignment (like MavlinkVisionPublisher)
        self._aligned = False
        self._pix_yaw0 = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="SITLVOBridge")
        self._thread.start()
        print("[SITLBridge] Stream thread started.")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        print(f"[SITLBridge] Stopped. Total messages sent: {self._sent}")

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _run(self):
        last_send = 0.0
        while not self._stop.is_set():
            now = time.time()
            if now - last_send < self._period:
                time.sleep(0.005)
                continue
            last_send = now

            # Poll ATTITUDE so we can align yaw (mirrors MavlinkVisionPublisher)
            att = self.master.recv_match(type="ATTITUDE", blocking=False)
            if att is not None and self._pix_yaw0 is None:
                self._pix_yaw0 = float(att.yaw)

            if self.vo.status != "TRACKING":
                continue

            pos, yaw_vis_deg = self.vo.pose()

            # Align once
            if not self._aligned and self._pix_yaw0 is not None:
                vo_yaw0 = math.radians(float(yaw_vis_deg))
                self._yaw_offset = self._pix_yaw0 - vo_yaw0
                self._aligned = True
                print(f"[SITLBridge] Yaw aligned. offset={math.degrees(self._yaw_offset):+.2f}°")

            # VO frame → NED (same mapping as vo_full_v3.py)
            x_vo, y_vo, z_vo = float(pos[0]), float(pos[1]), float(pos[2])
            ned = np.array([z_vo, x_vo, y_vo], dtype=np.float64)
            if self._aligned:
                c = math.cos(self._yaw_offset)
                s = math.sin(self._yaw_offset)
                ne = np.array([c*ned[0] - s*ned[1], s*ned[0] + c*ned[1]])
                ned[0], ned[1] = float(ne[0]), float(ne[1])

            yaw_ned = (self._pix_yaw0 if self._pix_yaw0 is not None
                       else math.radians(float(yaw_vis_deg)))

            self.master.mav.vision_position_estimate_send(
                _time_usec(self.boot_wall),
                float(ned[0]),   # x = North
                float(ned[1]),   # y = East
                float(ned[2]),   # z = Down
                0.0,             # roll
                0.0,             # pitch
                float(yaw_ned),  # yaw
            )
            self._sent += 1

            if self._sent % 150 == 0:   # log every 5s at 30 Hz
                print(f"[SITLBridge] VO→SITL | N={ned[0]:+.2f} E={ned[1]:+.2f} D={ned[2]:+.2f} "
                      f"yaw={math.degrees(yaw_ned):+.1f}° | msgs={self._sent}")


# ─── Main entry point ─────────────────────────────────────────────────────────

def main():
    """
    Spins up vo_full_v3.py in a background thread, then bridges its pose
    output into SITL. ArUco detection runs on the same RGB frame if enabled.
    """
    import importlib, sys, os

    # ── Import your VO system ────────────────────────────────────────────────
    # vo_full_v3.py must be in the same directory or on PYTHONPATH.
    try:
        vo_mod = importlib.import_module("vo_full_v3")
    except ModuleNotFoundError:
        print("[SITLBridge] ERROR: Could not import vo_full_v3. "
              "Make sure sitl_vo_bridge.py is in the same folder.")
        sys.exit(1)

    # ── Build a minimal VO pipeline without the full main() UI ──────────────
    # We reuse VO_LK and MavlinkVisionPublisher but skip the MAVLINK send
    # (the bridge handles that). DepthAI pipeline setup is copied from vo_full_v3.py.
    import depthai as dai
    import cv2
    import numpy as np

    FPS = vo_mod.FPS
    W, H = vo_mod.W, vo_mod.H
    RGB_SOCKET   = vo_mod.RGB_SOCKET
    LEFT_SOCKET  = vo_mod.LEFT_SOCKET
    RIGHT_SOCKET = vo_mod.RIGHT_SOCKET

    aruco_det = None
    if ENABLE_ARUCO:
        from aruco_detector import ArucoDetector
        aruco_det = ArucoDetector(valid_ids=ARUCO_VALID_IDS)

    bridge = None

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
            stereo.setDepthAlign(dai.StereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)

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

            calib = device.getCalibration()
            K = np.array(calib.getCameraIntrinsics(LEFT_SOCKET, W, H), dtype=np.float64)

            vo = vo_mod.VO_LK(K)
            loop_closure = vo_mod.LoopClosureORB() if vo_mod.ENABLE_LOOP else None

            # Start bridge after VO object exists
            bridge = SITLVOBridge(vo, aruco_detector=aruco_det)
            bridge.start()

            sync_q = sync.out.createOutputQueue()
            imu_q  = imu.out.createOutputQueue(maxSize=50, blocking=False)

            pipeline.start()
            t0 = time.time()
            frame_id = 0

            print("\n[SITLBridge] Running. Press Q in the camera window to quit.\n")

            while pipeline.isRunning():
                t_sec = time.time() - t0

                try:
                    imu_msgs = imu_q.tryGetAll()
                except Exception:
                    imu_msgs = []
                for msg in imu_msgs:
                    for pkt in msg.packets:
                        vo.update_imu(pkt.gyroscope.z)

                msg_group = sync_q.get()
                if msg_group is None:
                    continue

                rgb   = msg_group["rgb"].getCvFrame()
                gray  = msg_group["left"].getCvFrame()
                depth = msg_group["depth"].getFrame()

                if rgb is None or gray is None or depth is None:
                    continue

                vo.process(gray, depth)
                pos, yaw_vis = vo.pose()

                # Loop closure (optional, mirrors vo_full_v3.py)
                if loop_closure and vo.status == "TRACKING":
                    if frame_id % vo_mod.KEYFRAME_INTERVAL == 0:
                        loop_closure.add_keyframe(rgb, pos, frame_id, t_sec)
                    info = loop_closure.check_loop(rgb, pos, frame_id)
                    if info:
                        vo.apply_soft_correction(info["matched_pose"])
                        pos, yaw_vis = vo.pose()

                # ArUco detection
                detections = []
                if aruco_det is not None:
                    detections = aruco_det.process_frame(rgb, pos, t_sec)

                # Visualisation window
                vis = rgb.copy() if rgb is not None else np.zeros((H, W, 3), dtype=np.uint8)
                if aruco_det and detections:
                    vis = aruco_det.draw_overlays(vis, detections)

                cv2.putText(vis, f"SITL VO BRIDGE | status: {vo.status}", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(vis, f"N={pos[2]:+.2f}m  E={pos[0]:+.2f}m  D={pos[1]:+.2f}m",
                            (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2)
                cv2.putText(vis, f"tracked={vo.num_tracked}  inliers={vo.inliers}  "
                                 f"msgs_to_SITL={bridge._sent}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                aligned_txt = "aligned" if bridge._aligned else "aligning..."
                cv2.putText(vis, f"SITL yaw: {aligned_txt}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)

                cv2.imshow("SITL VO Bridge + ArUco", vis)
                frame_id += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()
    if bridge:
        bridge.stop()
    if aruco_det:
        aruco_det.close()
    print("[SITLBridge] Done.")


if __name__ == "__main__":
    main()