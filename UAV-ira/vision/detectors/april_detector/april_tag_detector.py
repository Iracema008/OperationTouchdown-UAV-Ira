# AprilTag 36h11 Detector
# Returns 3D position of tag in meters
#
# Assumptions:
# - Tag family: 36h11
# - Target tag ID: 67
# - Tag size: 20 cm (0.20 meters)

import cv2
from pupil_apriltags import Detector
import numpy as np
import depthai as dai


# Config / Adjust these as we test
TARGET_TAG_ID = 67
TAG_SIZE = 0.20     # 20 cm tag


class AprilTagDetector:

    def __init__(self, calibration_handler):
        """
        Initialize AprilTag detector using calibration
        from the OAK-D S2.
        """
        self.calibration_handler = calibration_handler 

        self.camera_matrix = None
        self.dist_coeffs = None

        self.FX = None
        self.FY = None
        self.CX = None
        self.CY = None

        print("[INFO] Detector initialized (intrinsics will be set on first frame)")

        self.detector = Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25
        )

    def _update_intrinsics(self, frame):
        """
        Update the camera intrinsics based on the current frame size.
        """
        h, w = frame.shape[:2]

        intrinsics = self.calibration_handler.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A,
            w,
            h
        )

        self.camera_matrix = np.array(intrinsics)

        self.FX = self.camera_matrix[0][0]
        self.FY = self.camera_matrix[1][1]
        self.CX = self.camera_matrix[0][2]
        self.CY = self.camera_matrix[1][2]

        # Load distortion coefficients as well (the detector needs them)
        self.dist_coeffs = np.array(
            self.calibration_handler.getDistortionCoefficients(
            dai.CameraBoardSocket.CAM_A
        )[:5]
)

        print(f"[INFO] Intrinsics updated for {w}x{h}")

    def get_tag_pose(self, frame):
        """
        Detect tag and return:

        (x, y, z) in meters relative to camera frame

        Camera Frame:
        x → right or left
        y → down or up
        z → forward or back

        Returns None if tag not detected.
        """

        if frame is None:
            return None

        if self.camera_matrix is None:
            self._update_intrinsics(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.undistort(
            gray,
            self.camera_matrix,
            self.dist_coeffs
        )

        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(self.FX, self.FY, self.CX, self.CY),
            tag_size=TAG_SIZE
        )

        if len(detections) == 0:
            return None

        # Search for our target and return the position
        for tag in detections:
            if tag.tag_id == TARGET_TAG_ID:
                t = tag.pose_t

                x = float(t[0][0])
                y = float(t[1][0])
                z = float(t[2][0])

                return x, y, z
        return None