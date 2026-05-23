# core/state_mp.py
""" Switched to multiprocessing safe UAV state """

import numpy as np
from multiprocessing import shared_memory, Lock, Event
from enum import Enum, auto
from dataclasses import dataclass
import time


class FlightMode(Enum):
    IDLE = auto()
    SCAN = auto()
    HOVER = auto()
    LAND = auto()
    ABORT = auto()


def create_shared_state():
    """
    Create all shared memory blocks and multiprocessing primitives.
    Call this ONCE from the main process before spawning children.
    
    Returns a dict of shared memory handles and locks/events.
    """
    # VIO position: [x, y, z, yaw, timestamp]
    shm_vio = shared_memory.SharedMemory(
        create=True, size=8 * 5, name="uav_vio"
    )
    
    # ArUco pose: [x, y, z, roll, pitch, yaw, marker_id, timestamp]
    shm_aruco = shared_memory.SharedMemory(
        create=True, size=8 * 8, name="uav_aruco"
    )
    
    # April pose: [x, y, z, roll, pitch, yaw, timestamp]
    shm_april = shared_memory.SharedMemory(
        create=True, size=8 * 7, name="uav_april"
    )
    
    # Pixhawk telemetry: [armed, battery_v, alt_m, heading_deg, timestamp]
    # armed stored as float (0.0 or 1.0) for array uniformity
    shm_telem = shared_memory.SharedMemory(
        create=True, size=8 * 5, name="uav_telem"
    )
    
    # Flight mode: single int
    shm_mode = shared_memory.SharedMemory(
        create=True, size=4, name="uav_mode"
    )
    
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
    """Close and unlink all shared memory. Call from main process on exit."""
    for key in ["shm_vio", "shm_aruco", "shm_april", "shm_telem", "shm_mode"]:
        shm = state_dict[key]
        shm.close()
        shm.unlink()


class UAVStateAccessor:
    """
    Accessor for shared UAV state from any process.
    Each process creates its own accessor instance.
    Do NOT pass this object between processes — it doesn't pickle.
    Instead, each process calls UAVStateAccessor() independently.
    """
    
    def __init__(self, lock, marker_confirmed, ugv_signal, hover_reached):
        self.lock = lock
        self.marker_confirmed = marker_confirmed
        self.ugv_signal = ugv_signal
        self.hover_reached = hover_reached
        
        # Map existing shared memory (created by main process)
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
        """Close shared memory handles (does not unlink — main process does that)."""
        self._shm_vio.close()
        self._shm_aruco.close()
        self._shm_april.close()
        self._shm_telem.close()
        self._shm_mode.close()
    
    # VIO
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
    
    # ArUco
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
    
    # April
    def set_april_pose(self, x, y, z):
        with self.lock:
            self._april[0] = x
            self._april[1] = y
            self._april[2] = z
            self._april[6] = time.time()
    
    def get_april_pose(self):
        with self.lock:
            return tuple(self._april[:3])
    
    # Telemetry
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
