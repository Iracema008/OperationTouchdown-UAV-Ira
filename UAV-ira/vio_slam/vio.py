"""  Visual-Inertial Odometry process. Reads frames from shared memory, computes position, writes to shared memory. """

import cv2
import numpy as np
import time
import math

from multiprocessing import shared_memory
from controls.connect import connect_UART3
from controls.busywait import delay_busywait
from core.state import UAVStateAccessor
from core.log import get_logger

logger = get_logger(__name__)


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
MIN_DRIFT_TO_CORRECT = 0.12
MAX_CORR_STEP_M = 1.0


# math for 3D coordinate transformation and a rotation from roll/pitch/yaw (reducing drift during tilted flight)
def clamp_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n <= 1e-9:
        return vec
    return vec * (max_norm / n)

def vo_step_to_ned(cam_x, cam_y, cam_z, roll_rad, pitch_rad, yaw_rad):
    ''' Converts camera-frame motion step to NED using full attitude rotation.'''

    #Improved from vo_full_v3, that only used  gyroscope z (yaw). Using full roll/pitch/yaw from the Pixhawk
    # IMU correctly rotates motion during tilted flight, reducing drift.

    # Camera frame to body frame
    body_x = -cam_y
    body_y =  cam_x
    body_z =  cam_z

    # Body frame to NED using rotation matrix from roll/pitch/yaw
    cr, sr = math.cos(roll_rad),  math.sin(roll_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cy, sy = math.cos(yaw_rad),   math.sin(yaw_rad)

    step_n = (cy*cp)*body_x + (cy*sp*sr - sy*cr)*body_y + (cy*sp*cr + sy*sr)*body_z
    step_e = (sy*cp)*body_x + (sy*sp*sr + cy*cr)*body_y + (sy*sp*cr - cy*sr)*body_z
    step_d = (-sp) *body_x + (cp*sr)             *body_y + (cp*cr)            *body_z

    return step_n, step_e, step_d

# still uses lk method with pnp 
class VO_LK:
    def __init__(self, K: np.ndarray):
        self.K    = K.astype(np.float64)
        self.dist = np.zeros((4, 1), dtype=np.float64)

        self.global_north = 0.0
        self.global_east  = 0.0
        self.global_down  = 0.0

        self.prev_gray  = None
        self.prev_depth = None
        self.prev_pts   = None

        self.status      = "INIT"
        self.num_tracked = 0
        self.frame_idx   = 0
        self._last_corr_wall = 0.0

    def _detect(self, gray):
        return cv2.goodFeaturesToTrack(
            gray, maxCorners=MAX_CORNERS,
            qualityLevel=QUALITY_LEVEL,
            minDistance=MIN_DISTANCE,
            blockSize=7, useHarrisDetector=False
        )

    def _reset_tracking(self, gray, depth_mm):
        self.prev_gray  = gray.copy()
        self.prev_depth = depth_mm.copy()
        self.prev_pts   = self._detect(gray)

    def process(self, gray, depth_mm, roll_rad, pitch_rad, yaw_rad):
        self.frame_idx += 1
        W, H = gray.shape[1], gray.shape[0]

        if (self.prev_gray is None
                or self.prev_depth is None
                or self.prev_pts is None):
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
        # There is a lot going on, but basically, for each tracked point, get the corresponding depth from the previous frame
        for (u0, v0), (u1, v1) in zip(prev_good, curr_good):
            x0, y0 = int(round(u0)), int(round(v0))
            if not (0 <= x0 < W and 0 <= y0 < H):
                continue
            z_m = float(self.prev_depth[y0, x0]) / 1000.0

            # Filters out points with invalid depth
            if z_m < DEPTH_MIN_M or z_m > DEPTH_MAX_M:
                continue
            obj_pts.append([(u0 - cx) * z_m / fx,
                             (v0 - cy) * z_m / fy, z_m])
            img_pts.append([u1, v1])

        # checks if we have enough valid 3D-2D correspondences to run PnP
        if len(obj_pts) < MIN_PNP_POINTS:
            # resets tracking to avoid drifting to bad points
            self.status = "DEPTH_FILTER"
            self._reset_tracking(gray, depth_mm)
            return

        obj_pts = np.asarray(obj_pts, dtype=np.float64)
        img_pts = np.asarray(img_pts, dtype=np.float64)

        # finally runs PnP to get the camera motion, and updates the global pos
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj_pts, img_pts, self.K, self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=3.0,
            confidence=0.999,
            iterationsCount=150
        )

        if not ok or inl is None or len(inl) < 9:
            self.status = "PNP_FAIL"
            self._reset_tracking(gray, depth_mm)
            return

        # rodrigues gives us the rotation from the current frame to the previous frame,
        # then we invert it to get the motion of the camera in the current frame
        R, _  = cv2.Rodrigues(rvec)
        t_inv = (-R.T @ tvec.reshape(3, 1)).flatten()

        step_n, step_e, step_d = vo_step_to_ned(
            float(t_inv[0]), float(t_inv[1]), float(t_inv[2]),
            roll_rad, pitch_rad, yaw_rad
        )

        self.global_north += step_n
        self.global_east  += step_e
        self.global_down  += step_d
        self.status = "TRACKING"

        if (self.frame_idx % REDETECT_EVERY) == 0:
            pts = self._detect(gray)
            self.prev_pts = (
                pts if pts is not None and len(pts) >= MIN_PNP_POINTS
                else curr_good.reshape(-1, 1, 2).astype(np.float32)
            )
        else:
            self.prev_pts = curr_good.reshape(-1, 1, 2).astype(np.float32)

        self.prev_gray  = gray.copy()
        self.prev_depth = depth_mm.copy()

    def apply_soft_correction(self, slam_target: np.ndarray):
        # gradually pulls towards the SLAM pos when the drift is big/ new correction
        now = time.time()
        if now - self._last_corr_wall < SOFT_CORR_COOLDOWN:
            return None
        if slam_target[0] < MIN_DRIFT_TO_CORRECT:
            return None

        corr = clamp_norm(slam_target[1:4], MAX_CORR_STEP_M) * SOFT_CORR_ALPHA
        self.global_north += corr[0]
        self.global_east  += corr[1]
        self.global_down  += corr[2]
        self._last_corr_wall = now
        return True

    def pose(self):
        return [self.global_north, self.global_east, self.global_down]

def run_vio_process(gray_frame_mutex, depth_frame_mutex,attitude_mutex, position_mutex, slam_trigger_mutex, lock, marker_confirmed, ugv_signal, hover_reached,cfg):
    logger.info("[VIO] Process starting")
    W, H = cfg.camera.width, cfg.camera.height

    # writes pos to uav_vio for mission to read
    state = UAVStateAccessor(lock, marker_confirmed, ugv_signal, hover_reached)

    # UART3 — sends vision_position_estimate to Pixhawk
    master_uart3 = connect_UART3()

    # VIO pipeline shared memory
    shm_gray         = shared_memory.SharedMemory(name="oak_gray")
    shm_depth        = shared_memory.SharedMemory(name="oak_depth")
    shm_calib        = shared_memory.SharedMemory(name="oak_calib")
    shm_attitude     = shared_memory.SharedMemory(name="attitude")
    shm_position     = shared_memory.SharedMemory(name="position")
    shm_slam_target  = shared_memory.SharedMemory(name="slam_target")
    shm_slam_trigger = shared_memory.SharedMemory(name="slam_trigger")

    shared_gray         = np.ndarray((H, W),    dtype=np.uint8,   buffer=shm_gray.buf)
    shared_depth        = np.ndarray((H, W),    dtype=np.uint16,  buffer=shm_depth.buf)
    shared_calib        = np.ndarray((3, 3),    dtype=np.float64, buffer=shm_calib.buf)
    shared_attitude     = np.ndarray((3,),      dtype=np.float64, buffer=shm_attitude.buf)
    shared_position     = np.ndarray((3,),      dtype=np.float64, buffer=shm_position.buf)
    shared_slam_target  = np.ndarray((4,),      dtype=np.float64, buffer=shm_slam_target.buf)
    shared_slam_trigger = np.ndarray((1,),      dtype=np.bool_,   buffer=shm_slam_trigger.buf)

    # Local buffers
    local_gray       = np.zeros((H, W), dtype=np.uint8)
    local_depth      = np.zeros((H, W), dtype=np.uint16)
    local_slam_target = np.zeros((4,),  dtype=np.float64)
    last_gray        = np.zeros((H, W), dtype=np.uint8)

    # Wait for broadcaster to write calibration, then read it
    logger.info("[VIO] Waiting for calibration from broadcaster...")
    time.sleep(1.0)
    with depth_frame_mutex:
        local_calib = shared_calib.copy()

    vo = VO_LK(K=local_calib)
    logger.info(
        f"[VIO] VO_LK initialised — "
        f"fx={local_calib[0,0]:.2f} fy={local_calib[1,1]:.2f} "
        f"cx={local_calib[0,2]:.2f} cy={local_calib[1,2]:.2f}"
    )

    try:
        while True:
            # Read grayscale
            with gray_frame_mutex:
                np.copyto(local_gray, shared_gray)

            # Skip duplicate frames
            if np.array_equal(local_gray, last_gray):
                delay_busywait(0.001)
                continue

            # Read depth + attitude
            with depth_frame_mutex:
                np.copyto(local_depth, shared_depth)
            with attitude_mutex:
                roll  = shared_attitude[0]
                pitch = shared_attitude[1]
                yaw   = shared_attitude[2]

            np.copyto(last_gray, local_gray)
            timestamp_usec = int(time.time() * 1e6)

            # Run VIO
            vo.process(local_gray, local_depth, roll, pitch, yaw)

            if vo.status == "TRACKING":
                # Apply SLAM correction if finding a loop closure
                with slam_trigger_mutex:
                    if shared_slam_trigger[0]:
                        np.copyto(local_slam_target, shared_slam_target)
                        vo.apply_soft_correction(local_slam_target)
                        shared_slam_trigger[0] = False

                # Send to Pixhawk (uart3)
                pos = vo.pose()

                # Write to VIO pipeline shared memory, SLAM reads this
                with position_mutex:
                    shared_position[:] = pos

                # Write to UAV state shared memory, mission reads this
                state.set_vio_position(
                    # n, e, d, yaw
                    float(pos[0]),
                    float(pos[1]),
                    float(pos[2]),
                    float(yaw)
                )

                # Send to Pixhawk via UART3
                master_uart3.mav.vision_position_estimate_send(
                    timestamp_usec,
                    pos[0], pos[1], pos[2],
                    0.0, 0.0, 0.0
                )

                logger.debug(
                    f"[VIO] N={pos[0]:.2f} E={pos[1]:.2f} D={pos[2]:.2f} "
                    f"tracked={vo.num_tracked}"
                )
            else:
                logger.debug(f"[VIO] {vo.status}")

    except KeyboardInterrupt:
        logger.info("[VIO] Interrupted")
    finally:
        shm_gray.close()
        shm_depth.close()
        shm_calib.close()
        shm_attitude.close()
        shm_position.close()
        shm_slam_target.close()
        shm_slam_trigger.close()
        state.close()
        logger.info("[VIO] Process exiting")