import cv2
import numpy as np
import time
import math
import multiprocessing as mp
from controls.busywait import delay_busywait
from vio_slam.broadcaster import broadcaster
from multiprocessing import shared_memory
from pymavlink import mavutil


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
    while angle_rad > math.pi: angle_rad -= 2.0 * math.pi
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

    def add_keyframe(self, frame: np.ndarray, pose_xyz: np.ndarray, frame_id: int, t_sec: float):
        small = self._prep(frame)
        if small is None: return
        
        _, des = self.orb.detectAndCompute(small, None)
        if des is None or len(des) == 0: return

        self.keyframes.append({
            "id": frame_id,
            "t_sec": t_sec,
            "des": des,
            "pose": pose_xyz.copy(),
        })

        if len(self.keyframes) > MAX_KEYFRAMES:
            self.keyframes.pop(0)

    def check_loop(self, frame: np.ndarray, current_pose_xyz: np.ndarray, frame_id: int):
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

        return (drift_mag, float(drift_vec[0]), float(drift_vec[1]), float(drift_vec[2]))
def slam(rgb_frame_mutex, attitude_mutex, position_mutex, slam_enabled_mutex, slam_trigger_mutex):
    W, H = 640, 400
    
    shm_rgb = shared_memory.SharedMemory(name="oak_rgb")
    shm_attitude = shared_memory.SharedMemory(name="attitude")
    shm_position = shared_memory.SharedMemory(name="position")
    shm_slam_target = shared_memory.SharedMemory(name="slam_target")
    shm_slam_trigger = shared_memory.SharedMemory(name="slam_trigger")
    shm_slam_enabled = shared_memory.SharedMemory(name="slam_enabled")

    shared_rgb = np.ndarray((H, W, 3), dtype=np.uint8, buffer=shm_rgb.buf)
    shared_attitude = np.ndarray((3,), dtype=np.float64, buffer=shm_attitude.buf)
    shared_position = np.ndarray((3,), dtype=np.float64, buffer=shm_position.buf)
    shared_slam_target = np.ndarray((4,), dtype=np.float64, buffer=shm_slam_target.buf)
    shared_slam_trigger = np.ndarray((1,), dtype=np.bool_, buffer=shm_slam_trigger.buf)
    shared_slam_enabled = np.ndarray((1,), dtype=np.bool_, buffer=shm_slam_enabled.buf)

    last_processed_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    local_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    local_position = np.zeros((3,), dtype=np.float64)

    loop = LoopClosureORB()
    t0 = time.time()
    
    last_kf_pos = None
    last_kf_yaw = None
    slam_frame_id = 0
    with slam_enabled_mutex:
        shared_slam_enabled[0] = True
    print("SLAM setup complete and running")

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

        if last_kf_pos is None or last_kf_yaw is None:
            loop.add_keyframe(local_rgb, local_position, slam_frame_id, t_sec)
            last_kf_pos = local_position.copy()
            last_kf_yaw = live_yaw
        else:
            yaw_changed = abs(wrap_rad_pi(live_yaw - last_kf_yaw))
            if yaw_changed >= KEYFRAME_MIN_YAW_RAD:
                loop.add_keyframe(local_rgb, local_position, slam_frame_id, t_sec)
                last_kf_pos = local_position.copy()
                last_kf_yaw = live_yaw
            else:
                dist_moved = float(np.linalg.norm(local_position - last_kf_pos))
                if dist_moved >= KEYFRAME_MIN_DIST_M:
                    loop.add_keyframe(local_rgb, local_position, slam_frame_id, t_sec)
                    last_kf_pos = local_position.copy()
                    last_kf_yaw = live_yaw
        # 2. LOOP CLOSURE SEARCH
        info = loop.check_loop(local_rgb, local_position, slam_frame_id)
        if info is not None:
            with slam_trigger_mutex:
                shared_slam_target[:] = info
                shared_slam_trigger[0] = True
            # Overwrite baseline so teleport doesn't instantly trigger a false spatial keyframe
            last_kf_pos = info["matched_pose"].copy()
def test_latency_slam(rgb_frame_mutex, attitude_mutex, position_mutex, slam_enabled_mutex, slam_trigger_mutex):
    W, H = 640, 400
    count = 0
    average_ms = 0.0
    
    shm_rgb = shared_memory.SharedMemory(name="oak_rgb")
    shm_attitude = shared_memory.SharedMemory(name="attitude")
    shm_position = shared_memory.SharedMemory(name="position")
    shm_slam_target = shared_memory.SharedMemory(name="slam_target")
    shm_slam_trigger = shared_memory.SharedMemory(name="slam_trigger")
    shm_slam_enabled = shared_memory.SharedMemory(name="slam_enabled")

    shared_rgb = np.ndarray((H, W, 3), dtype=np.uint8, buffer=shm_rgb.buf)
    shared_attitude = np.ndarray((3,), dtype=np.float64, buffer=shm_attitude.buf)
    shared_position = np.ndarray((3,), dtype=np.float64, buffer=shm_position.buf)
    shared_slam_target = np.ndarray((4,), dtype=np.float64, buffer=shm_slam_target.buf)
    shared_slam_trigger = np.ndarray((1,), dtype=np.bool_, buffer=shm_slam_trigger.buf)
    shared_slam_enabled = np.ndarray((1,), dtype=np.bool_, buffer=shm_slam_enabled.buf)

    last_processed_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    local_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    local_position = np.zeros((3,), dtype=np.float64)

    loop = LoopClosureORB()
    t0 = time.time()
    
    last_kf_pos = None
    last_kf_yaw = None
    slam_frame_id = 0
    with slam_enabled_mutex:
        shared_slam_enabled[0] = True
    print("SLAM setup complete and running")

    while True:
        start_time = time.perf_counter()
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

        if last_kf_pos is None or last_kf_yaw is None:
            loop.add_keyframe(local_rgb, local_position, slam_frame_id, t_sec)
            last_kf_pos = local_position.copy()
            last_kf_yaw = live_yaw
        else:
            yaw_changed = abs(wrap_rad_pi(live_yaw - last_kf_yaw))
            if yaw_changed >= KEYFRAME_MIN_YAW_RAD:
                loop.add_keyframe(local_rgb, local_position, slam_frame_id, t_sec)
                last_kf_pos = local_position.copy()
                last_kf_yaw = live_yaw
            else:
                dist_moved = float(np.linalg.norm(local_position - last_kf_pos))
                if dist_moved >= KEYFRAME_MIN_DIST_M:
                    loop.add_keyframe(local_rgb, local_position, slam_frame_id, t_sec)
                    last_kf_pos = local_position.copy()
                    last_kf_yaw = live_yaw
        # 2. LOOP CLOSURE SEARCH
        info = loop.check_loop(local_rgb, local_position, slam_frame_id)
        if info is not None:
            with slam_trigger_mutex:
                shared_slam_target[:] = info
                shared_slam_trigger[0] = True
            # Update the baseline by adding the drift vector
            last_kf_pos += np.array([info[1], info[2], info[3]])
            end_time = time.perf_counter()
            time_elapsed_ms = (end_time - start_time) * 1000.0
            if time_elapsed_ms > 5.0:
                average_ms += time_elapsed_ms
                count += 1
            if count >= 8:
                print(f"Average SLAM: {average_ms / count:.3f} ms")
                count = 0
                average_ms = 0.0
def main(position_mutex):
    shm_position = shared_memory.SharedMemory(name="position")
    shared_position = np.ndarray((3,), dtype=np.float64, buffer=shm_position.buf)
    local_position = np.zeros((3,), dtype=np.float64)

    print("Main Connected to Shared Memory. Listening for NED coordinates...")
    while True:
        with position_mutex:
            np.copyto(local_position, shared_position)

        print(f"North:{local_position[0]:+.2f}m | East: {local_position[1]:+.2f}m | Down: {local_position[2]:+.2f}m")
        time.sleep(0.25) 
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    W, H = 640, 400
    RGB_BYTES = W * H * 3
    GRAY_BYTES = W * H
    DEPTH_BYTES = W * H * 2 
    CALIB_BYTES = 3 * 3 * 8 
    ATTITUDE_BYTES = 3 * 8 
    POSITION_BYTES = 3 * 8 
    LOCAL_POSITION_NED_BYTES = 3 * 8
    BOOL_BYTES = 1
    TARGET_BYTES = 4 * 8

    print("SLAM tester allocating shared memory...")
    shm_rgb = shared_memory.SharedMemory(create=True, size=RGB_BYTES, name="oak_rgb")
    shm_gray = shared_memory.SharedMemory(create=True, size=GRAY_BYTES, name="oak_gray")
    shm_depth = shared_memory.SharedMemory(create=True, size=DEPTH_BYTES, name="oak_depth")
    shm_calib = shared_memory.SharedMemory(create=True, size=CALIB_BYTES, name="oak_calib")
    shm_attitude = shared_memory.SharedMemory(create=True, size=ATTITUDE_BYTES, name="attitude")
    shm_position = shared_memory.SharedMemory(create=True, size=POSITION_BYTES, name="position")
    shm_local_position_ned = shared_memory.SharedMemory(create=True, size=LOCAL_POSITION_NED_BYTES, name="local_position_ned")
    shm_slam_enabled = shared_memory.SharedMemory(create=True, size=BOOL_BYTES, name="slam_enabled")
    shm_slam_target = shared_memory.SharedMemory(create=True, size=TARGET_BYTES, name="slam_target")
    shm_slam_trigger = shared_memory.SharedMemory(create=True, size=BOOL_BYTES, name="slam_trigger")
    print("SLAM tester finished allocating shared memory...")

    rgb_frame_mutex = mp.Lock()
    gray_frame_mutex = mp.Lock()
    depth_frame_mutex = mp.Lock()
    attitude_mutex = mp.Lock()
    position_mutex = mp.Lock()
    local_position_ned_mutex = mp.Lock()
    slam_trigger_mutex = mp.Lock()
    slam_enabled_mutex = mp.Lock()

    from vio_slam.vio import vio
    broadcaster_process = mp.Process(target=broadcaster, args=(rgb_frame_mutex, gray_frame_mutex, depth_frame_mutex, attitude_mutex, local_position_ned_mutex,))
    vio_process = mp.Process(target=vio, args=(gray_frame_mutex, depth_frame_mutex, attitude_mutex, position_mutex, slam_trigger_mutex,))
    slam_process = mp.Process(target=test_latency_slam, args=(rgb_frame_mutex, attitude_mutex, position_mutex, slam_enabled_mutex, slam_trigger_mutex,))
    main_process = mp.Process(target=main, args=(position_mutex,))

    try:
        broadcaster_process.start()
        time.sleep(3)
        vio_process.start()
        time.sleep(3)
        slam_process.start()
        time.sleep(3)
        main_process.start()
        time.sleep(3)
        main_process.join()
        
    except KeyboardInterrupt:
        print("SLAM tester caught keyboard interrupt. Shutting down...")
    finally:
        broadcaster_process.terminate()
        vio_process.terminate()
        slam_process.terminate()
        main_process.terminate()
        
        broadcaster_process.join()
        vio_process.join()
        slam_process.join()
        main_process.join()

        print("SLAM tester cleaning up shared memory...")
        shm_rgb.close()
        shm_rgb.unlink()
        shm_gray.close()
        shm_gray.unlink()
        shm_depth.close()
        shm_depth.unlink()
        shm_calib.close()
        shm_calib.unlink()
        shm_attitude.close()
        shm_attitude.unlink()
        shm_position.close()
        shm_position.unlink()
        shm_local_position_ned.close()
        shm_local_position_ned.unlink()
        shm_slam_enabled.close()
        shm_slam_enabled.unlink()
        shm_slam_target.close()
        shm_slam_target.unlink()
        shm_slam_trigger.close()
        shm_slam_trigger.unlink()
        print("SLAM tester processes terminated safely.")