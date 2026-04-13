# Stationary Landing Script
#
# For now this just creates its own connection to the camera and
# we will need to integrate the controller into our actual main loop later.

import depthai as dai
from Detectors.april_tag_detector import AprilTagDetector
from UAV.PixhawkController.stationary_landing_controller import StationaryLandingController

# Pixhawk connection port and baudrate (need to adjust)
CONNECTION_STRING = "/dev/ttyACM0"
BAUDRATE = 57600

# Landing threshold (if we are closer than this to the tag, we will land)
LANDING_THRESHOLD = 0.3   # Adjust as we test (this is the distance

# Start up the camera and the OAK-D pipeline
pipeline = dai.Pipeline()

cam = pipeline.createColorCamera()
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

xout = pipeline.createXLinkOut()
xout.setStreamName("rgb")
cam.video.link(xout.input)

# Start main logic
with dai.Device(pipeline) as device:

    print("[INFO] OAK-D Started")

    # Load calibration
    calibration = device.readCalibration()

    # Create detector
    april_tag_detector = AprilTagDetector(calibration)

    # Create the Stationary Landing Controller (connects to Pixhawk)
    controller = StationaryLandingController(CONNECTION_STRING, BAUDRATE)

    # For testing purposes, get into the correct modes and take off to 3 meters before we start the landing logic
    controller.test_arm_and_takeoff()
    q_rgb = device.getOutputQueue("rgb")

    while True:

        in_rgb = q_rgb.get()
        frame = in_rgb.getCvFrame()

        pose = april_tag_detector.get_tag_pose(frame)

        if pose is None:
            print("[WARN] Tag not detected, hovering...")
            controller.send_velocity(0, 0, 0)  # hold
            continue

        cam_x, cam_y, cam_z = pose
        print(f"[INFO] Tag Position (Camera): X={cam_x:.2f}, Y={cam_y:.2f}, Z={cam_z:.2f} m")

        # Convert to body frame
        body_x, body_y, body_z = controller.convert_camera_to_body_frame(cam_x, cam_y, cam_z)
        print(f"[INFO] Tag Position (Body): {body_x:.2f}, {body_y:.2f}, {body_z:.2f}")

        # If we are close enough, land
        if body_z < LANDING_THRESHOLD:
            controller.land_and_disarm(body_z)
            break
        # Not close enough, keep sending velocity commands to move towards the tag
        controller.adjust_velocity_and_send(body_x, body_y, body_z)