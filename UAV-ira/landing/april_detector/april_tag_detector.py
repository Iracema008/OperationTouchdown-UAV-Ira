# AprilTag 36h11 Detector
# Returns 3D position of tag in meters
#
# Assumptions:
# - Tag family: 36h11
# - Target tag ID: 0
# - Tag size: 20 cm (0.20 meters)

import cv2
import apriltag
import numpy as np
import depthai as dai


# Config / Adjust these as we test
TARGET_TAG_ID = 0
TAG_SIZE = 0.20     # 20 cm tag


class AprilTagDetector:

    def __init__(self, calibration_handler):
        """
        Initialize AprilTag detector using calibration
        from the OAK-D S2.
        """
        # Load the camera intrinsics from the OAK-D calibration handler
        #
        # NOTE: This assumes we are using full resolution RGB stream for detection.
        intrinsics = calibration_handler.getCameraIntrinsics(
            dai.CameraBoardSocket.RGB
        )

        self.camera_matrix = np.array(intrinsics)

        self.FX = self.camera_matrix[0][0]
        self.FY = self.camera_matrix[1][1]
        self.CX = self.camera_matrix[0][2]
        self.CY = self.camera_matrix[1][2]

        print("[INFO] Camera Intrinsics Loaded")

        # Load distortion coefficients as well (the detector needs them)
        self.dist_coeffs = np.array(
            calibration_handler.getDistortionCoefficients(
                dai.CameraBoardSocket.RGB
            )
        )

        print("[INFO] Distortion Coefficients Loaded")

       # Configure to just look for our family of tags
        options = apriltag.DetectorOptions(
            families="tag36h11"
        )

        self.detector = apriltag.Detector(options)


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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

                x = float(t[0])
                y = float(t[1])
                z = float(t[2])

                return x, y, z
        return None