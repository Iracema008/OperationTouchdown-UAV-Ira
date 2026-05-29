import cv2
import numpy as np
import time
import math
import multiprocessing as mp
from controls.busywait import delay_busywait
from controls.connect import connect_UART3
from vio_slam.broadcaster import broadcaster
from multiprocessing import shared_memory
from pymavlink import mavutil

# -----------------------
# VIO Settings
# -----------------------
MAX_CORNERS = 300
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 10
LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 3
MIN_PNP_POINTS = 9
DEPTH_MIN_M = 0.08
DEPTH_MAX_M = 18.0
REDETECT_EVERY = 10
SOFT_CORR_ALPHA = 0.25 
SOFT_CORR_COOLDOWN = 0.75 
MIN_DRIFT_TO_CORRECT_M = 0.12 
MAX_CORR_STEP_M = 1.0 

def clamp_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n <= 1e-9 or n <= max_norm: return vec
    return vec * (max_norm / n)
def vo_step_to_ned(cam_x, cam_y, cam_z, roll_rad, pitch_rad, yaw_rad):
    body_x = -cam_y  
    body_y = cam_x   
    body_z = cam_z   

    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)

    R00 = cy * cp
    R01 = cy * sp * sr - sy * cr
    R02 = cy * sp * cr + sy * sr

    R10 = sy * cp
    R11 = sy * sp * sr + cy * cr
    R12 = sy * sp * cr - cy * sr

    R20 = -sp
    R21 = cp * sr
    R22 = cp * cr

    step_n = R00 * body_x + R01 * body_y + R02 * body_z
    step_e = R10 * body_x + R11 * body_y + R12 * body_z
    step_d = R20 * body_x + R21 * body_y + R22 * body_z

    return step_n, step_e, step_d
class VO_LK:
    def __init__(self, K: np.ndarray):
        self.K = K.astype(np.float64)
        self.dist = np.zeros((4, 1), dtype=np.float64)
        
        self.global_north = 0.0
        self.global_east = 0.0
        self.global_down = 0.0

        self.prev_gray = None
        self.prev_depth = None
        self.prev_pts = None

        self.status = "INIT"
        self.num_tracked = 0
        self.frame_idx = 0
        self._last_corr_wall = 0.0

    def _detect(self, gray):
        return cv2.goodFeaturesToTrack(
            gray, maxCorners=MAX_CORNERS, qualityLevel=QUALITY_LEVEL,
            minDistance=MIN_DISTANCE, blockSize=7, useHarrisDetector=False
        )

    def _reset_tracking(self, gray, depth_mm):
        self.prev_gray = gray.copy()
        self.prev_depth = depth_mm.copy()
        self.prev_pts = self._detect(gray)

    def process(self, gray, depth_mm, roll_rad, pitch_rad, yaw_rad):
        self.frame_idx += 1
        W, H = gray.shape[1], gray.shape[0]

        if self.prev_gray is None or self.prev_depth is None or self.prev_pts is None:
            self._reset_tracking(gray, depth_mm)
            self.status = "WARMUP"
            return

        next_pts, st, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None,
            winSize=LK_WIN_SIZE, maxLevel=LK_MAX_LEVEL,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        if next_pts is None or st is None:
            self.status = "LK_FAIL"
            self._reset_tracking(gray, depth_mm) 
            return

        st = st.reshape(-1)
        prev_good = self.prev_pts[st == 1].reshape(-1, 2)
        curr_good = next_pts[st == 1].reshape(-1, 2)
        self.num_tracked = len(prev_good)

        if self.num_tracked < MIN_PNP_POINTS:
            self.status = f"LOW_TRACK({self.num_tracked})"
            self._reset_tracking(gray, depth_mm) 
            return

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        obj_pts, img_pts = [], []
        for (u0, v0), (u1, v1) in zip(prev_good, curr_good):
            x0, y0 = int(round(u0)), int(round(v0))
            if not (0 <= x0 < W and 0 <= y0 < H): continue

            z_m = float(self.prev_depth[y0, x0]) / 1000.0
            if z_m < DEPTH_MIN_M or z_m > DEPTH_MAX_M: continue

            X = (u0 - cx) * z_m / fx
            Y = (v0 - cy) * z_m / fy
            obj_pts.append([X, Y, z_m])
            img_pts.append([u1, v1])

        if len(obj_pts) < MIN_PNP_POINTS:
            self.status = "DEPTH_FILTER"
            self._reset_tracking(gray, depth_mm) 
            return

        obj_pts = np.asarray(obj_pts, dtype=np.float64)
        img_pts = np.asarray(img_pts, dtype=np.float64)

        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj_pts, img_pts, self.K, self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=3.0,
            confidence=0.999, iterationsCount=150
        )
        
        if not ok or inl is None or len(inl) < 9:
            self.status = "PNP_FAIL"
            self._reset_tracking(gray, depth_mm) 
            return

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)

        t_inv = (-R.T @ t).flatten()

        step_n, step_e, step_d = vo_step_to_ned(
            float(t_inv[0]), float(t_inv[1]), float(t_inv[2]), 
            roll_rad, pitch_rad, yaw_rad
        )

        self.global_north += step_n
        self.global_east += step_e
        self.global_down += step_d

        self.status = "TRACKING"

        if (self.frame_idx % REDETECT_EVERY) == 0:
            pts = self._detect(gray)
            self.prev_pts = pts if pts is not None and len(pts) >= MIN_PNP_POINTS else curr_good.reshape(-1, 1, 2).astype(np.float32)
        else:
            self.prev_pts = curr_good.reshape(-1, 1, 2).astype(np.float32)

        self.prev_gray = gray.copy()
        self.prev_depth = depth_mm.copy()

    def apply_soft_correction(self, slam_target_data: np.ndarray):
        now = time.time()
        if now - self._last_corr_wall < SOFT_CORR_COOLDOWN: return None
        if slam_target_data[0] < MIN_DRIFT_TO_CORRECT_M: return None

        step = clamp_norm(slam_target_data[1:4], MAX_CORR_STEP_M)
        corr = step * SOFT_CORR_ALPHA

        self.global_north += corr[0]
        self.global_east += corr[1]
        self.global_down += corr[2]
        
        self._last_corr_wall = now
        return True

    def pose(self):
        return [self.global_north, self.global_east, self.global_down]
def vio(gray_frame_mutex, depth_frame_mutex, attitude_mutex, position_mutex, slam_trigger_mutex):
    master_uart3 = connect_UART3()
    W, H = 640, 400
    
    shm_gray = shared_memory.SharedMemory(name="oak_gray")
    shm_depth = shared_memory.SharedMemory(name="oak_depth")
    shm_calib = shared_memory.SharedMemory(name="oak_calib")
    shm_attitude = shared_memory.SharedMemory(name="attitude")
    shm_position = shared_memory.SharedMemory(name="position")
    shm_slam_target = shared_memory.SharedMemory(name="slam_target")
    shm_slam_trigger = shared_memory.SharedMemory(name="slam_trigger")

    shared_calib = np.ndarray((3, 3), dtype=np.float64, buffer=shm_calib.buf)
    shared_gray = np.ndarray((H, W), dtype=np.uint8, buffer=shm_gray.buf)
    shared_depth = np.ndarray((H, W), dtype=np.uint16, buffer=shm_depth.buf)
    shared_attitude = np.ndarray((3,), dtype=np.float64, buffer=shm_attitude.buf)
    shared_position = np.ndarray((3,), dtype=np.float64, buffer=shm_position.buf)
    shared_slam_target = np.ndarray((4,), dtype=np.float64, buffer=shm_slam_target.buf)
    shared_slam_trigger = np.ndarray((1,), dtype=np.bool_, buffer=shm_slam_trigger.buf)

    local_calib = np.zeros((3, 3), dtype=np.float64)
    local_gray = np.zeros((H, W), dtype=np.uint8)
    local_depth = np.zeros((H, W), dtype=np.uint16)
    local_slam_target = np.zeros((4,), dtype=np.float64)
    last_processed_gray = np.zeros((H, W), dtype=np.uint8)

    with depth_frame_mutex:
        np.copyto(local_calib, shared_calib)
    
    vo = VO_LK(K=local_calib.copy())
    print("VIO setup complete and running")

    while True:
        with gray_frame_mutex:
             np.copyto(local_gray, shared_gray)
        if np.array_equal(local_gray, last_processed_gray):
            delay_busywait(0.001)
            continue

        with depth_frame_mutex:
             np.copyto(local_depth, shared_depth)
        with attitude_mutex:
            live_roll = shared_attitude[0]
            live_pitch = shared_attitude[1]
            live_yaw = shared_attitude[2]
        np.copyto(last_processed_gray, local_gray)
        timestamp_usec = int(time.time() * 1e6)

        vo.process(local_gray, local_depth, live_roll, live_pitch, live_yaw)
        if vo.status == "TRACKING":
            # Check if Background SLAM has found a loop closure
            with slam_trigger_mutex:
                if shared_slam_trigger[0]:
                    np.copyto(local_slam_target, shared_slam_target)
                    vo.apply_soft_correction(local_slam_target)
                    shared_slam_trigger[0] = False # Acknowledge receipt
            # Send to Pixhawk
            pos = vo.pose()
            with position_mutex:
                shared_position[:] = pos
            master_uart3.mav.vision_position_estimate_send(timestamp_usec, pos[0], pos[1], pos[2], 0.0, 0.0, 0.0)
        else:
            print(f"VIO Tracking Lost: {vo.status}")
def test_latency_vio(gray_frame_mutex, depth_frame_mutex, attitude_mutex, position_mutex, slam_trigger_mutex):
    master_uart3 = connect_UART3()
    W, H = 640, 400
    count = 0
    average_ms = 0.0
    
    shm_gray = shared_memory.SharedMemory(name="oak_gray")
    shm_depth = shared_memory.SharedMemory(name="oak_depth")
    shm_calib = shared_memory.SharedMemory(name="oak_calib")
    shm_attitude = shared_memory.SharedMemory(name="attitude")
    shm_position = shared_memory.SharedMemory(name="position")
    shm_slam_target = shared_memory.SharedMemory(name="slam_target")
    shm_slam_trigger = shared_memory.SharedMemory(name="slam_trigger")

    shared_calib = np.ndarray((3, 3), dtype=np.float64, buffer=shm_calib.buf)
    shared_gray = np.ndarray((H, W), dtype=np.uint8, buffer=shm_gray.buf)
    shared_depth = np.ndarray((H, W), dtype=np.uint16, buffer=shm_depth.buf)
    shared_attitude = np.ndarray((3,), dtype=np.float64, buffer=shm_attitude.buf)
    shared_position = np.ndarray((3,), dtype=np.float64, buffer=shm_position.buf)
    shared_slam_target = np.ndarray((4,), dtype=np.float64, buffer=shm_slam_target.buf)
    shared_slam_trigger = np.ndarray((1,), dtype=np.bool_, buffer=shm_slam_trigger.buf)

    local_calib = np.zeros((3, 3), dtype=np.float64)
    local_gray = np.zeros((H, W), dtype=np.uint8)
    local_depth = np.zeros((H, W), dtype=np.uint16)
    local_slam_target = np.zeros((4,), dtype=np.float64)
    last_processed_gray = np.zeros((H, W), dtype=np.uint8)

    with depth_frame_mutex:
        np.copyto(local_calib, shared_calib)
    
    vo = VO_LK(K=local_calib.copy())
    print("VIO setup complete and running")

    while True:
        start_time = time.perf_counter()
        with gray_frame_mutex:
             np.copyto(local_gray, shared_gray)
        if np.array_equal(local_gray, last_processed_gray):
            delay_busywait(0.001)
            continue

        with depth_frame_mutex:
             np.copyto(local_depth, shared_depth)
        with attitude_mutex:
            live_roll = shared_attitude[0]
            live_pitch = shared_attitude[1]
            live_yaw = shared_attitude[2]
        np.copyto(last_processed_gray, local_gray)
        timestamp_usec = int(time.time() * 1e6)

        vo.process(local_gray, local_depth, live_roll, live_pitch, live_yaw)
        end_time = time.perf_counter()
        if vo.status == "TRACKING":
            # Check if Background SLAM has found a loop closure
            with slam_trigger_mutex:
                if shared_slam_trigger[0]:
                    np.copyto(local_slam_target, shared_slam_target)
                    vo.apply_soft_correction(local_slam_target)
                    shared_slam_trigger[0] = False # Acknowledge receipt
            # Send to Pixhawk
            pos = vo.pose()
            with position_mutex:
                shared_position[:] = pos
            master_uart3.mav.vision_position_estimate_send(timestamp_usec, pos[0], pos[1], pos[2], 0.0, 0.0, 0.0)
            average_ms += (end_time - start_time) * 1000.0
            count += 1
            if count >= 32:
                print(f"Average VIO: {average_ms / count:.3f} ms")
                count = 0
                average_ms = 0.0
        else:
            print(f"VIO Tracking Lost: {vo.status}")
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

    print("VIO tester allocating shared memory...")
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
    print("VIO tester finished allocating shared memory...")

    rgb_frame_mutex = mp.Lock()
    gray_frame_mutex = mp.Lock()
    depth_frame_mutex = mp.Lock()
    attitude_mutex = mp.Lock()
    position_mutex = mp.Lock()
    local_position_ned_mutex = mp.Lock()
    slam_trigger_mutex = mp.Lock()
    slam_enabled_mutex = mp.Lock()

    from vio_slam.slam import slam
    broadcaster_process = mp.Process(target=broadcaster, args=(rgb_frame_mutex, gray_frame_mutex, depth_frame_mutex, attitude_mutex, local_position_ned_mutex,))
    vio_process = mp.Process(target=test_latency_vio, args=(gray_frame_mutex, depth_frame_mutex, attitude_mutex, position_mutex, slam_trigger_mutex,))
    slam_process = mp.Process(target=slam, args=(rgb_frame_mutex, attitude_mutex, position_mutex, slam_enabled_mutex, slam_trigger_mutex,))
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
        print("VIO tester caught keyboard interrupt. Shutting down...")
    finally:
        broadcaster_process.terminate()
        vio_process.terminate()
        slam_process.terminate()
        main_process.terminate()

        broadcaster_process.join()
        vio_process.join()
        slam_process.join()
        main_process.join()

        print("VIO tester cleaning up shared memory...")
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
        print("VIO tester processes terminated safely.")