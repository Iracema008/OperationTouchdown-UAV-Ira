import time
import depthai as dai

from vision.detectors.april_detector.april_tag_detector import AprilTagDetector
from landing.pixhawk_controller import (
    StationaryLandingController
)

CONNECTION_STRING = "/dev/serial0"
BAUDRATE = 57600

LANDING_THRESHOLD = 0.3
TAKEOFF_ALTITUDE = 6  # meters

HOVER_TIMEOUT = 7.0
SEARCH_TIMEOUT = 15.0

# Start the pipeline / camera

with dai.Device() as device:

    print("[INFO] OAK-D Started")

    # Load calibration for AprilTag detection
    calibration = device.getCalibration()

    april_tag_detector = AprilTagDetector(calibration)

    controller = StationaryLandingController(
        CONNECTION_STRING,
        BAUDRATE
    )

    # Create the pipeline

    with dai.Pipeline(device) as pipeline:
        cam_rgb = pipeline.create(dai.node.Camera)
        cam_rgb.build(dai.CameraBoardSocket.CAM_A)

        rgb_out = cam_rgb.requestOutput(
            size=(640, 480),
            type=dai.ImgFrame.Type.BGR888p,
            fps=30
        )

        q_rgb = rgb_out.createOutputQueue(
            maxSize=4,
            blocking=False
        )

        pipeline.start()

        print("[INFO] Pipeline started")
        print("[INFO] Initiating takeoff sequence...")

        controller.change_flight_mode("GUIDED")
        controller.arm_motors()
        controller.takeoff_to_altitude(TAKEOFF_ALTITUDE)

        last_tag_time = time.time()
        while pipeline.isRunning():
            in_rgb = q_rgb.tryGet()

            if in_rgb is None:
                time.sleep(0.01)
                continue
            
            # Get the current frame
            frame = in_rgb.getCvFrame()

            # Get the x y z from the april tag
            pose = april_tag_detector.get_tag_pose(frame)

            # Is tag lost?
            if pose is None:
                time_lost = time.time() - last_tag_time

                if time_lost < HOVER_TIMEOUT:
                    print(
                        f"[WARN] Tag lost for "
                        f"{time_lost:.1f}s. Hovering..."
                    )
                    controller.send_velocity(0, 0, 0)

                elif time_lost < SEARCH_TIMEOUT:
                    print(
                        f"[WARN] Tag lost for "
                        f"{time_lost:.1f}s. Ascending..."
                    )
                    controller.send_velocity(0, 0, -0.2)

                else:
                    print(
                        "[CRITICAL] Tag lost too long. "
                        "Blind landing..."
                    )
                    controller.stationary_landing()
                    time.sleep(5)
                    controller.disarm_motors()
                    break

                continue

            # tag detected, reset timer
            last_tag_time = time.time()
            cam_x, cam_y, cam_z = pose
            print(
                f"[INFO] Camera Frame | "
                f"X={cam_x:.2f}, "
                f"Y={cam_y:.2f}, "
                f"Z={cam_z:.2f}"
            )

            # camera to body frame conversion
            body_x, body_y, body_z = (
                controller.convert_camera_to_body_frame(
                    cam_x,
                    cam_y,
                    cam_z
                )
            )
            print(
                f"[INFO] Body Frame | "
                f"X={body_x:.2f}, "
                f"Y={body_y:.2f}, "
                f"Z={body_z:.2f}"
            )

            # Check landing conditions
            if (
                abs(body_x) < 0.10 and
                abs(body_y) < 0.10 and
                body_z < LANDING_THRESHOLD
            ):
                print("[INFO] Landing conditions reached")
                controller.stationary_landing()
                time.sleep(5)
                controller.disarm_motors()
                break

            # track the tag by sending velocity commands
            controller.adjust_velocity_and_send(
                body_x,
                body_y,
                body_z
            )
            # Small sleep to stabilize loop timing
            time.sleep(0.05)