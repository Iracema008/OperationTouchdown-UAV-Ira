# core/config.py
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class CameraConfig:
    # OAK-D S2 — RVC2, depthai v3
    # rgb cam
    color_socket: str = "CAM_A"
    # left & right mono
    left_socket:  str = "CAM_B"
    right_socket: str = "CAM_C"

    color_fps: int = 30
    mono_fps:  int = 60               # mono cams support higher fps
    color_resolution: str = "THE_1080_P"   # dai.ColorCameraProperties.SensorResolution
    mono_resolution:  str = "THE_400_P"    # dai.MonoCameraProperties.SensorResolution

    depth_enabled: bool = True
    imu_enabled:   bool = True        # S2 has built-in IMU (BNO085)
    stereo_confidence_threshold: int = 200  # 0-255, higher = more filtering

    # Pull these from: python -c "import depthai as dai; d=dai.Device(); print(d.readCalibration().getCameraIntrinsics(dai.CameraBoardSocket.CAM_A))"
    fx: float = 1371.0
    fy: float = 1371.0
    cx: float = 960.0
    cy: float = 540.0


@dataclass
class ArucoConfig:
    target_marker_id: int = 0
    marker_size_m: float = 0.2             # physical size in meters
    dictionary: str = "DICT_4X4_50"        # OpenCV aruco dict name
    detection_fps: int = 30
    min_consecutive_detections: int = 5    # confirm marker only after N frames


@dataclass
class AprilTagConfig:
    target_family: str = "tag36h11"
    tag_size_m: float = 0.16               # physical size in meters


@dataclass
class PixhawkConfig:
    connection_string: str = "/dev/ttyAMA0"   # swap to "udp:127.0.0.1:14550" for SITL
    baud_rate: int = 57600
    hover_altitude_m: float = 3.0
    land_speed_ms:    float = 0.3             # descent speed m/s
    alignment_threshold_m: float = 0.15      # how close to tag center before descending
    landing_threshold: float = 1.5


@dataclass
class CommsConfig:
    serial_port: str = "/dev/ttyUSB0"
    baud_rate: int = 9600
    ugv_ready_message: str = "LAND"


@dataclass
class SLAMConfig:
    publish_rate_hz: int = 30
    csv_output_dir:  str = "logs/"
    csv_enabled:     bool = True


@dataclass
class Config:
    mode: str = "scan"                        # "scan" or "land"
    camera:  CameraConfig  = field(default_factory=CameraConfig)
    aruco:   ArucoConfig   = field(default_factory=ArucoConfig)
    april:   AprilTagConfig = field(default_factory=AprilTagConfig)
    pixhawk: PixhawkConfig = field(default_factory=PixhawkConfig)
    comms:   CommsConfig   = field(default_factory=CommsConfig)
    slam:    SLAMConfig    = field(default_factory=SLAMConfig)


def load_config(mode: Optional[str] = None) -> Config:
    cfg = Config()
    if mode:
        cfg.mode = mode
    if os.getenv("PIXHAWK_PORT"):
        cfg.pixhawk.connection_string = os.getenv("PIXHAWK_PORT")
    if os.getenv("UAV_MODE"):
        cfg.mode = os.getenv("UAV_MODE")
    return cfg