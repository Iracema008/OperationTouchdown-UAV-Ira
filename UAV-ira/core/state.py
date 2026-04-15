# core/state.py
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import time


class FlightMode(Enum):
    IDLE    = auto()
    SCAN    = auto()
    HOVER   = auto()
    LAND    = auto()
    ABORT   = auto()


@dataclass
class Pose3D:
    x: float = 0.0   # meters, body-frame forward
    y: float = 0.0   # meters, body-frame right
    z: float = 0.0   # meters, down positive (NED)
    roll:  float = 0.0  # radians
    pitch: float = 0.0
    yaw:   float = 0.0
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class PixhawkTelemetry:
    armed: bool = False
    mode: str = "UNKNOWN"
    battery_voltage: float = 0.0
    relative_altitude: float = 0.0  # meters AGL
    heading: float = 0.0            # degrees
    timestamp: float = field(default_factory=time.monotonic)


class UAVState:
    """
    Central shared state for all UAV subsystems.
    All public methods are thread-safe.
    Use Events to wait on state transitions — never spin-poll.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # --- Vision ---
        self._aruco_pose: Optional[Pose3D] = None
        self._aruco_marker_id: Optional[int] = None

        # --- Landing ---
        self._april_pose: Optional[Pose3D] = None

        # --- SLAM / VIO ---
        self._vio_position: Optional[Pose3D] = None

        # --- Pixhawk telemetry ---
        self._telemetry: PixhawkTelemetry = PixhawkTelemetry()

        # --- Flight mode ---
        self._flight_mode: FlightMode = FlightMode.IDLE

        # --- Events (non-blocking wait primitives) ---
        # Set once when aruco marker is confirmed; never cleared (latch)
        self.marker_confirmed = threading.Event()
        # Set by comms thread when UGV signals ready-to-land
        self.ugv_signal = threading.Event()
        # Set when drone reaches hover setpoint
        self.hover_reached = threading.Event()

    # ------------------------------------------------------------------ #
    #  Aruco / vision                                                      #
    # ------------------------------------------------------------------ #

    def set_aruco_pose(self, pose: Pose3D, marker_id: int) -> None:
        with self._lock:
            self._aruco_pose = pose
            self._aruco_marker_id = marker_id
        self.marker_confirmed.set()   # latch — first valid detection fires this

    def get_aruco_pose(self) -> tuple[Optional[Pose3D], Optional[int]]:
        with self._lock:
            return self._aruco_pose, self._aruco_marker_id

    # ------------------------------------------------------------------ #
    #  April tag / landing                                                 #
    # ------------------------------------------------------------------ #

    def set_april_pose(self, pose: Pose3D) -> None:
        with self._lock:
            self._april_pose = pose

    def get_april_pose(self) -> Optional[Pose3D]:
        with self._lock:
            return self._april_pose

    # ------------------------------------------------------------------ #
    #  VIO / SLAM                                                          #
    # ------------------------------------------------------------------ #

    def set_vio_position(self, pose: Pose3D) -> None:
        with self._lock:
            self._vio_position = pose

    def get_vio_position(self) -> Optional[Pose3D]:
        with self._lock:
            return self._vio_position

    # ------------------------------------------------------------------ #
    #  Pixhawk telemetry                                                   #
    # ------------------------------------------------------------------ #

    def set_telemetry(self, telem: PixhawkTelemetry) -> None:
        with self._lock:
            self._telemetry = telem

    def get_telemetry(self) -> PixhawkTelemetry:
        with self._lock:
            return self._telemetry

    # ------------------------------------------------------------------ #
    #  Flight mode                                                         #
    # ------------------------------------------------------------------ #

    def set_flight_mode(self, mode: FlightMode) -> None:
        with self._lock:
            self._flight_mode = mode

    def get_flight_mode(self) -> FlightMode:
        with self._lock:
            return self._flight_mode

    # ------------------------------------------------------------------ #
    #  Debug snapshot (safe to call any time)                              #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "flight_mode":        self._flight_mode.name,
                "aruco_marker_id":    self._aruco_marker_id,
                "marker_confirmed":   self.marker_confirmed.is_set(),
                "ugv_signal":         self.ugv_signal.is_set(),
                "hover_reached":      self.hover_reached.is_set(),
                "vio_position":       self._vio_position,
                "telemetry":          self._telemetry,
            }