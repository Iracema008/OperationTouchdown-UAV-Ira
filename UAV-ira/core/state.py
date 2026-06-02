""" Multiprocessing safe UAV state with shared memory. """

import numpy as np
import time
from multiprocessing import shared_memory, Lock, Event
from enum import Enum, auto


class FlightMode(Enum):
    IDLE = auto()
    SCAN = auto()
    HOVER = auto()
    LAND = auto()
    ABORT = auto()


def create_shared_state():
    """ Creates state shared memory blocks, calls once from main process before spawning children. """
    # VIO position: [x, y, z, yaw, timestamp]
    shm_vio = shared_memory.SharedMemory(create=True, size=8 * 5, name="uav_vio")

    # ArUco pose: [x, y, z, roll, pitch, yaw, marker_id, timestamp]
    shm_aruco = shared_memory.SharedMemory(create=True, size=8 * 8, name="uav_aruco")

    # April pose: [x, y, z, roll, pitch, yaw, timestamp]
    shm_april = shared_memory.SharedMemory(create=True, size=8 * 7, name="uav_april")

    # Pixhawk telemetry: [armed, battery_v, alt_m, heading_deg, timestamp]
    shm_telem = shared_memory.SharedMemory(create=True, size=8 * 5, name="uav_telem")

    # Flight mode: single int
    shm_mode = shared_memory.SharedMemory( create=True, size=4, name="uav_mode" )

    # Initialize all to zero
    np.ndarray((5,), dtype=np.float64, buffer=shm_vio.buf)[:] = 0
    np.ndarray((8,), dtype=np.float64, buffer=shm_aruco.buf)[:] = 0
    np.ndarray((7,), dtype=np.float64, buffer=shm_april.buf)[:] = 0
    np.ndarray((5,), dtype=np.float64, buffer=shm_telem.buf)[:] = 0
    np.ndarray((1,), dtype=np.int32, buffer=shm_mode.buf)[0] = FlightMode.IDLE.value

    return {
        "shm_vio": shm_vio,
        "shm_aruco": shm_aruco,
        "shm_april": shm_april,
        "shm_telem": shm_telem,
        "shm_mode": shm_mode,
        "lock": Lock(),
        "marker_confirmed": Event(),
        "ugv_signal": Event(),
        "hover_reached": Event(),
    }


def cleanup_shared_state(state_dict):
    """Close and unlink UAV state shared memory. Main process only."""
    for key in ["shm_vio", "shm_aruco", "shm_april", "shm_telem", "shm_mode"]:
        shm = state_dict[key]
        shm.close()
        shm.unlink()


def create_vio_pipeline_state(W: int = 640, H: int = 400):
    # 1. Set the multiprocessing start method to 'spawn' this is so that when a new process is started it
    #    doesn't inherit the memory of the parent process, and instead creates its own memory space and also
    #    its own python interpreter. Effectively isolating the processes from each other except for the shared memory.

    RGB_BYTES = W * H * 3
    GRAY_BYTES = W * H
    DEPTH_BYTES = W * H * 2
    CALIB_BYTES = 3 * 3 * 8
    ATTITUDE_BYTES = 3 * 8
    POSITION_BYTES = 3 * 8 
    LOCAL_POS_BYTES = 3 * 8
    BOOL_BYTES = 1
    TARGET_BYTES = 4 * 8


    # 2. Create the shared memory for RGB, gray, depth, camera calibration matrix, and attitude
    shm_rgb = shared_memory.SharedMemory(create=True, size=RGB_BYTES, name="oak_rgb")
    shm_gray = shared_memory.SharedMemory(create=True, size=GRAY_BYTES, name="oak_gray")
    shm_depth = shared_memory.SharedMemory(create=True, size=DEPTH_BYTES, name="oak_depth")
    shm_calib = shared_memory.SharedMemory(create=True, size=CALIB_BYTES, name="oak_calib")
    shm_attitude = shared_memory.SharedMemory(create=True, size=ATTITUDE_BYTES, name="attitude")
    shm_position = shared_memory.SharedMemory(create=True, size=POSITION_BYTES, name="position")
    shm_local_position_ned = shared_memory.SharedMemory(create=True, size=LOCAL_POS_BYTES, name="local_position_ned")

    shm_slam_enabled = shared_memory.SharedMemory(create=True, size=BOOL_BYTES, name="slam_enabled")
    shm_slam_target = shared_memory.SharedMemory(create=True, size=TARGET_BYTES, name="slam_target")
    shm_slam_trigger = shared_memory.SharedMemory(create=True, size=BOOL_BYTES, name="slam_trigger")

    # Initialize to zero
    np.ndarray((H, W, 3), dtype=np.uint8, buffer=shm_rgb.buf)[:] = 0
    np.ndarray((H, W), dtype=np.uint8, buffer=shm_gray.buf)[:] = 0
    np.ndarray((H, W), dtype=np.uint16, buffer=shm_depth.buf)[:] = 0
    np.ndarray((3, 3), dtype=np.float64, buffer=shm_calib.buf)[:] = 0
    np.ndarray((3,), dtype=np.float64, buffer=shm_attitude.buf)[:] = 0
    np.ndarray((3,), dtype=np.float64, buffer=shm_position.buf)[:] = 0
    np.ndarray((3,), dtype=np.float64, buffer=shm_local_position_ned.buf)[:] = 0
    np.ndarray((1,), dtype=np.bool_, buffer=shm_slam_enabled.buf)[0] = False
    np.ndarray((4,), dtype=np.float64, buffer=shm_slam_target.buf)[:] = 0
    np.ndarray((1,), dtype=np.bool_, buffer=shm_slam_trigger.buf)[0] = False

    return {
        "shm_rgb": shm_rgb,
        "shm_gray": shm_gray,
        "shm_depth": shm_depth,
        "shm_calib": shm_calib,
        "shm_attitude": shm_attitude,
        "shm_position": shm_position,
        "shm_local_position_ned": shm_local_position_ned,
        "shm_slam_enabled": shm_slam_enabled,
        "shm_slam_target": shm_slam_target,
        "shm_slam_trigger": shm_slam_trigger,

        # Mutexes for each block
        "rgb_frame_mutex": Lock(),
        "gray_frame_mutex": Lock(),
        "depth_frame_mutex": Lock(),
        "attitude_mutex": Lock(),
        "position_mutex": Lock(),
        "local_position_ned_mutex": Lock(),
        "slam_trigger_mutex": Lock(),
        "slam_enabled_mutex": Lock(),
    }


def cleanup_vio_pipeline_state(vio_dict):
    """Close and unlink VIO pipeline shared memory. Main process only."""
    shm_keys = [
        "shm_rgb", "shm_gray", "shm_depth", "shm_calib",
        "shm_attitude", "shm_position", "shm_local_position_ned",
        "shm_slam_enabled", "shm_slam_target", "shm_slam_trigger",
    ]
    for key in shm_keys:
        shm = vio_dict[key]
        shm.close()
        shm.unlink()


class UAVStateAccessor:
    """
    Accessor for shared UAV state from any process.
    Each process creates its own instance. Do NOT pickle/pass between processes.
    """

    def __init__(self, lock, marker_confirmed, ugv_signal, hover_reached):
        self.lock = lock
        self.marker_confirmed = marker_confirmed
        self.ugv_signal = ugv_signal
        self.hover_reached = hover_reached

        self._shm_vio = shared_memory.SharedMemory(name="uav_vio")
        self._shm_aruco = shared_memory.SharedMemory(name="uav_aruco")
        self._shm_april = shared_memory.SharedMemory(name="uav_april")
        self._shm_telem = shared_memory.SharedMemory(name="uav_telem")
        self._shm_mode = shared_memory.SharedMemory(name="uav_mode")

        self._vio = np.ndarray((5,), dtype=np.float64, buffer=self._shm_vio.buf)
        self._aruco = np.ndarray((8,), dtype=np.float64, buffer=self._shm_aruco.buf)
        self._april = np.ndarray((7,), dtype=np.float64, buffer=self._shm_april.buf)
        self._telem = np.ndarray((5,), dtype=np.float64, buffer=self._shm_telem.buf)
        self._mode = np.ndarray((1,), dtype=np.int32, buffer=self._shm_mode.buf)

    def close(self):
        self._shm_vio.close()
        self._shm_aruco.close()
        self._shm_april.close()
        self._shm_telem.close()
        self._shm_mode.close()

    def set_vio_position(self, x, y, z, yaw):
        with self.lock:
            self._vio[0] = x
            self._vio[1] = y
            self._vio[2] = z
            self._vio[3] = yaw
            self._vio[4] = time.time()

    def get_vio_position(self):
        with self.lock:
            return tuple(self._vio[:4]), self._vio[4]

    def set_aruco_pose(self, x, y, z, marker_id):
        with self.lock:
            self._aruco[0] = x
            self._aruco[1] = y
            self._aruco[2] = z
            self._aruco[6] = float(marker_id)
            self._aruco[7] = time.time()
        self.marker_confirmed.set()

    def get_aruco_pose(self):
        with self.lock:
            marker_id = int(self._aruco[6]) if self._aruco[6] != 0 else None
            return (self._aruco[0], self._aruco[1], self._aruco[2]), marker_id

    def set_april_pose(self, x, y, z):
        with self.lock:
            self._april[0] = x
            self._april[1] = y
            self._april[2] = z
            self._april[6] = time.time()

    def get_april_pose(self):
        with self.lock:
            return tuple(self._april[:3])

    def set_telemetry(self, armed, battery_v, alt_m, heading_deg):
        with self.lock:
            self._telem[0] = 1.0 if armed else 0.0
            self._telem[1] = battery_v
            self._telem[2] = alt_m
            self._telem[3] = heading_deg
            self._telem[4] = time.time()

    def get_telemetry(self):
        with self.lock:
            return {
                "armed": bool(self._telem[0]),
                "battery_v": self._telem[1],
                "alt_m": self._telem[2],
                "heading_deg": self._telem[3],
                "timestamp": self._telem[4],
            }

    # Flight mode
    def set_flight_mode(self, mode: FlightMode):
        with self.lock:
            self._mode[0] = mode.value

    def get_flight_mode(self) -> FlightMode:
        with self.lock:
            return FlightMode(self._mode[0])