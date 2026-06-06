from dataclasses import dataclass, field
from typing import Optional, List
import os


@dataclass
class CameraConfig:
    # OAK-D S2, RVC2, depthai v3
    color_socket: str = "CAM_A"
    left_socket:  str = "CAM_B"
    right_socket: str = "CAM_C"

    color_fps: int = 30
    mono_fps:  int = 60

    # keep at 640x400 to match broadcaster from broadcaster

    width:  int = 640
    height: int = 400

    mono_resolution: str = "THE_400_P"

    depth_enabled: bool = True
    imu_enabled:   bool = True
    stereo_confidence_threshold: int = 200

    # got these values from readCalibration().getCameraIntrinsics() w/boardsocket cam_a res
    fx: float = 576.18
    fy: float = 575.93
    cx: float = 311.67
    cy: float = 214.92


@dataclass
class VideoConfig:
    width:int= 640
    height: int = 400
    show_video:bool = False
    # fov_h =2 * atan(640/ (2 * 576.57))= 58.096110396
    # fov_v = 2 * atan(400 / (2 * 575.93)) ≈ 38.30059057
    fov_h:float = 58.09
    fov_v:float = 38.30


@dataclass
class DetectorConfig:
    detector_type: str = "CV2"


@dataclass
class ArucoConfig:
    target_marker_id: List[int] = field(default_factory=lambda: [0, 2])
    marker_size_m: float = 0.254
    dictionary: str = "DICT_6X6_250"
    detection_fps: int = 30
    min_consecutive_detections: int = 3


@dataclass
class AprilTagConfig:
    target_family: str = "tag36h11"
    tag_size_m: float = 0.20



# comment and uncomment here for sitl and regular
@dataclass
class PixhawkConfig:
    # comment out udp for sitl, keep serial0 for flight
    #connection_string: str =  "udp:0.0.0.0:14550"
    connection_string: str = "/dev/serial0"
    baud_rate: int = 57600
    hover_altitude_m: float = 3.0
    land_speed_ms: float = 0.3
    alignment_threshold_m:float = 0.15
    landing_threshold: float = 0.3


@dataclass
class UARTConfig:
    #connect uart2 ports from broadcaster
    # for the simulation this port 
    broadcaster_port: str = "/dev/ttyAMA2"
    #uart3 for vio
    vio_port: str = "/dev/ttyAMA3"
    baud_rate: int = 57600


@dataclass
class CommsConfig:
    serial_port: str = "/dev/ttyUSB0"
    baud_rate: int = 9600
    ugv_ready_message: str = "LAND"


@dataclass
class SLAMConfig:
    publish_rate_hz: int = 30
    csv_output_dir:str = "logs/"
    csv_enabled: bool = True


@dataclass
class Config:
    mode: str = "scan"
    camera:   CameraConfig   = field(default_factory=CameraConfig)
    video:    VideoConfig    = field(default_factory=VideoConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    aruco:    ArucoConfig    = field(default_factory=ArucoConfig)
    april:    AprilTagConfig = field(default_factory=AprilTagConfig)
    pixhawk:  PixhawkConfig  = field(default_factory=PixhawkConfig)
    uart:     UARTConfig = field(default_factory=UARTConfig)
    comms:    CommsConfig = field(default_factory=CommsConfig)
    slam:     SLAMConfig = field(default_factory=SLAMConfig)


def load_config(mode: Optional[str] = None) -> Config:
    cfg = Config()
    if mode:
        cfg.mode = mode
    if os.getenv("PIXHAWK_PORT"):
        cfg.pixhawk.connection_string = os.getenv("PIXHAWK_PORT")
    if os.getenv("UAV_MODE"):
        cfg.mode = os.getenv("UAV_MODE")
    return cfg
