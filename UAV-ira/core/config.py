# core/config.py
from dataclasses import dataclass, field
from typing import Optional, List
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

    # Resolution set to 640x480 to match vision pipeline request size.
    # Intrinsics (fx, fy, cx, cy) below are calibrated at this resolution.
    # If you change this, re-run calibration:
    #   python -c "import depthai as dai; d=dai.Device();
    #   print(d.readCalibration().getCameraIntrinsics(
    #   dai.CameraBoardSocket.CAM_A, 640, 480))"
    width:  int = 640
    height: int = 480

    mono_resolution: str = "THE_400_P"  # dai.MonoCameraProperties.SensorResolution

    depth_enabled: bool = True
    imu_enabled:   bool = True          # S2 has built-in IMU (BNO085)
    stereo_confidence_threshold: int = 200  # 0-255, higher = more filtering

    # Intrinsics calibrated at 640x480 — pull from your device with:
    # python -c "import depthai as dai; d=dai.Device();
    #   print(d.readCalibration().getCameraIntrinsics(
    #   dai.CameraBoardSocket.CAM_A, 640, 480))"
    # Values below are placeholders — replace with your actual calibration
    fx: float = 518.57    # approx half of 1371 scaled to 640px width
    fy: float = 518.34
    cx: float = 312.50    # center of 640px frame
    cy: float = 253.42    # center of 480px frame


@dataclass
class VideoConfig:
    # Must match CameraConfig width/height above
    width:      int  = 640
    height:     int  = 480
    show_video: bool = False   # set True for ground testing with monitor


@dataclass
class DetectorConfig:
    # Must match the key in AvailableDetectors enum
    detector_type: str = "CV2"


@dataclass
class ArucoConfig:
    # List of valid landing target IDs — drone will ignore any other marker
    target_marker_id: List[int] = field(default_factory=lambda: [3, 7])
    marker_size_m: float = 0.2              # physical size in meters
    dictionary: str = "DICT_6X6_250"        # must match Cv2Detector
    detection_fps: int = 30
    # Note: lawnmower_search.py FIELD_CONFIG also has confirm_detections=3
    # keep these in sync
    min_consecutive_detections: int = 3


@dataclass
class AprilTagConfig:
    target_family: str = "tag36h11"
    tag_size_m: float = 0.20               # physical size in meters


@dataclass
class PixhawkConfig:
    connection_string: str = "/dev/serial0"   # swap to "udp:127.0.0.1:14550" for SITL
    baud_rate: int = 57600                    # correct for Pi UART to Pixhawk
    hover_altitude_m: float = 3.0
    land_speed_ms:    float = 0.3             # descent speed m/s
    alignment_threshold_m: float = 0.15      # how close to tag center before descending
    landing_threshold: float = 0.3           # metres — was 1.5 which was way too high


@dataclass
class CommsConfig:
    # UGV serial comms — separate port from Pixhawk
    serial_port: str = "/dev/ttyUSB0"
    baud_rate: int = 9600        # fine for short string messages like "LAND"
                                 # bump to 115200 if sending larger payloads
    ugv_ready_message: str = "LAND"


@dataclass
class SLAMConfig:
    publish_rate_hz: int = 30
    csv_output_dir:  str = "logs/"
    csv_enabled:     bool = True


@dataclass
class Config:
    mode: str = "scan"                         # "scan" or "land"
    camera:   CameraConfig   = field(default_factory=CameraConfig)
    video:    VideoConfig    = field(default_factory=VideoConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    aruco:    ArucoConfig    = field(default_factory=ArucoConfig)
    april:    AprilTagConfig = field(default_factory=AprilTagConfig)
    pixhawk:  PixhawkConfig  = field(default_factory=PixhawkConfig)
    comms:    CommsConfig    = field(default_factory=CommsConfig)
    slam:     SLAMConfig     = field(default_factory=SLAMConfig)


def load_config(mode: Optional[str] = None) -> Config:
    cfg = Config()
    if mode:
        cfg.mode = mode
    # Environment variable overrides — useful for SITL testing without
    # editing this file:
    #   export PIXHAWK_PORT="udp:127.0.0.1:14550"
    #   export UAV_MODE="scan"
    if os.getenv("PIXHAWK_PORT"):
        cfg.pixhawk.connection_string = os.getenv("PIXHAWK_PORT")
    if os.getenv("UAV_MODE"):
        cfg.mode = os.getenv("UAV_MODE")
    return cfg