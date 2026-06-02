""" SLAM proccess for loop closure, runs orb, drift correction, and writes to shared memory """

import cv2
import numpy as np
import time
import math

from multiprocessing import shared_memory
from pymavlink import mavutil
from controls.busywait import delay_busywait
from core.log import get_logger

logger = get_logger(__name__)

KEYFRAME_MIN_DIST_M = 0.08
KEYFRAME_MIN_YAW_RAD = 0.17
LOOP_CHECK_INTERVAL = 0.6
MIN_LOOP_SEPARATION = 15
MATCH_THRESHOLD = 45
MAX_KEYFRAMES = 600
MAX_MATCH_CANDIDATES = 325
ORB_NFEATURES = 400
ORB_SCALE = 0.5


def wrap_rad_pi(angle_rad: float) -> float:
    while angle_rad >  math.pi: angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi: angle_rad += 2.0 * math.pi
    return angle_rad

class LoopClosureORB:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=ORB_NFEATURES)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.keyframes = []
        self.last_check_wall = 0.0

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame is None: return None
        if len(frame.shape) == 2: return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        gray = self._to_gray(frame)
        if gray is None: return None
        if ORB_SCALE != 1.0:
            new_w = max(1, int(gray.shape[1] * ORB_SCALE))
            new_h = max(1, int(gray.shape[0] * ORB_SCALE))
            return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return gray

    def add_keyframe(self, frame, pose_xyz, frame_id, t_sec):
        small = self._prep(frame)
        if small is None: return
        _, des = self.orb.detectAndCompute(small, None)
        if des is None or len(des) == 0: return
        self.keyframes.append({
            "id": frame_id, "t_sec": t_sec,
            "des": des, "pose": pose_xyz.copy(),
        })
        if len(self.keyframes) > MAX_KEYFRAMES:
            self.keyframes.pop(0)

    def check_loop(self, frame, current_pose_xyz, frame_id):
        now = time.time()
        if now - self.last_check_wall < LOOP_CHECK_INTERVAL: return None
        self.last_check_wall = now
        if len(self.keyframes) < (MIN_LOOP_SEPARATION + 1): return None

        small = self._prep(frame)
        if small is None: return None
        _, des = self.orb.detectAndCompute(small, None)
        if des is None or len(des) == 0: return None

        candidates = self.keyframes[:-MIN_LOOP_SEPARATION]
        if len(candidates) > MAX_MATCH_CANDIDATES:
            candidates = candidates[-MAX_MATCH_CANDIDATES:]

        best = None
        best_score = 0
        for kf in candidates:
            if frame_id - kf["id"] < MIN_LOOP_SEPARATION: continue
            try:
                matches = self.bf.match(kf["des"], des)
                score = len(matches)
                if score > best_score:
                    best_score = score
                    best = kf
            except Exception:
                continue

        if best is None or best_score < MATCH_THRESHOLD: return None

        drift_vec = best["pose"] - current_pose_xyz
        drift_mag = float(np.linalg.norm(drift_vec))
        return (drift_mag, float(drift_vec[0]),
                float(drift_vec[1]), float(drift_vec[2]))


def run_slam_process(rgb_frame_mutex, attitude_mutex, position_mutex, slam_enabled_mutex, slam_trigger_mutex, cfg):
    """ Reads RGB from shared memory, builds ORB keyframes, detects loop closures, and writes drift corrections for VIO to apply. """
    logger.info("[SLAM] Process starting")
    W, H = cfg.camera.width, cfg.camera.height

    shm_rgb = shared_memory.SharedMemory(name="oak_rgb")
    shm_attitude = shared_memory.SharedMemory(name="attitude")
    shm_position = shared_memory.SharedMemory(name="position")
    shm_slam_target  = shared_memory.SharedMemory(name="slam_target")
    shm_slam_trigger = shared_memory.SharedMemory(name="slam_trigger")
    shm_slam_enabled = shared_memory.SharedMemory(name="slam_enabled")

    shared_rgb = np.ndarray((H, W, 3), dtype=np.uint8, buffer=shm_rgb.buf)
    shared_attitude = np.ndarray((3,),dtype=np.float64, buffer=shm_attitude.buf)
    shared_position = np.ndarray((3,), dtype=np.float64, buffer=shm_position.buf)
    shared_slam_target = np.ndarray((4,), dtype=np.float64, buffer=shm_slam_target.buf)
    shared_slam_trigger = np.ndarray((1,), dtype=np.bool_, buffer=shm_slam_trigger.buf)
    shared_slam_enabled = np.ndarray((1,), dtype=np.bool_, buffer=shm_slam_enabled.buf)

    last_processed_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    local_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    local_position = np.zeros((3,), dtype=np.float64)

    loop = LoopClosureORB()
    t0 = time.time()

    last_kf_pos  = None
    last_kf_yaw  = None
    slam_frame_id = 0

    # Signal VIO that SLAM is ready
    with slam_enabled_mutex:
        shared_slam_enabled[0] = True

    logger.info("[SLAM] Setup complete and Loop closure running")

    try:
        while True:
            with slam_enabled_mutex:
                slam_enabled = bool(shared_slam_enabled[0])
            if not slam_enabled:
                time.sleep(2.5)
                continue

            with rgb_frame_mutex:
                np.copyto(local_rgb, shared_rgb)

            if np.array_equal(local_rgb, last_processed_rgb):
                delay_busywait(0.005)
                continue

            np.copyto(last_processed_rgb, local_rgb)
            slam_frame_id += 1
            t_sec = time.time() - t0

            with attitude_mutex:
                live_yaw = shared_attitude[2]
            with position_mutex:
                np.copyto(local_position, shared_position)

            # Keyframe decision
            if last_kf_pos is None or last_kf_yaw is None:
                loop.add_keyframe(local_rgb, local_position, slam_frame_id, t_sec)
                last_kf_pos = local_position.copy()
                last_kf_yaw = live_yaw
            else:
                yaw_changed = abs(wrap_rad_pi(live_yaw - last_kf_yaw))
                dist_moved  = float(np.linalg.norm(local_position - last_kf_pos))

                if (yaw_changed >= KEYFRAME_MIN_YAW_RAD
                        or dist_moved >= KEYFRAME_MIN_DIST_M):
                    loop.add_keyframe(
                        local_rgb, local_position, slam_frame_id, t_sec
                    )
                    last_kf_pos = local_position.copy()
                    last_kf_yaw = live_yaw

            # 2. LOOP CLOSURE SEARCH
            info = loop.check_loop(local_rgb, local_position, slam_frame_id)
            if info is not None:
                with slam_trigger_mutex:
                    shared_slam_target[:] = info
                    shared_slam_trigger[0] = True

                # Overwrite baseline so teleport doesn't instantly trigger a false spatial keyframe
                last_kf_pos += np.array([info[1], info[2], info[3]])
                logger.info(
                    f"[SLAM] Loop closure — drift={info[0]:.3f}m "
                    f"corr=({info[1]:.3f},{info[2]:.3f},{info[3]:.3f})"
                )

    except KeyboardInterrupt:
        logger.info("[SLAM] Interrupted")
    finally:
        shm_rgb.close()
        shm_attitude.close()
        shm_position.close()
        shm_slam_target.close()
        shm_slam_trigger.close()
        shm_slam_enabled.close()
        logger.info("[SLAM] Process exiting")