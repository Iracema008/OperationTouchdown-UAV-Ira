import time
import depthai as dai

from Detectors.april_tag_detector import AprilTagDetector
from PixhawkController.stationary_landing_controller import (
    StationaryLandingController
)

CONNECTION_STRING = "/dev/serial0"
BAUDRATE = 57600

LANDING_THRESHOLD = 0.4
TAKEOFF_ALTITUDE = 3.0  # meters

HOVER_TIMEOUT = 3.0
SEARCH_TIMEOUT = 7.0

COAST_BOOST = 1.3

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
        is_escaping_ground = False
        escape_target_z = 0.0
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
                COAST_BOOST = 1.3  

                # --- 1. EMERGENCY GROUND ESCAPE FAILSAFE ---
                # If we are already in escape-climb mode, or if we just breached the 0.2m deck
                if is_escaping_ground or controller.prev_z < 0.2:
                    if not is_escaping_ground:
                        is_escaping_ground = True
                        escape_target_z = controller.prev_z + 0.5
                        print(
                            f"[CRITICAL] Dangerously low ({controller.prev_z:.2f}m) while blind! "
                            f"Forcing 0.5m escape climb to target {escape_target_z:.2f}m..."
                        )

                    # Keep climbing until we cross our target altitude threshold
                    if controller.prev_z < escape_target_z:
                        # Ascend firmly (-0.3 m/s) while continuing to match the tag's predicted speed
                        controller.send_velocity(controller.last_vx * COAST_BOOST, controller.last_vy * COAST_BOOST, -0.5)
                        continue  # Bypass all other timers; focus entirely on escaping the floor
                    else:
                        print(f"[INFO] Ground escape successful! Reached {controller.prev_z:.2f}m. Resuming search.")
                        is_escaping_ground = False  # Reset flag to return to normal tracking

                # --- 2. LOW ALTITUDE LAND FAILSAFE ---
                # If we are low (below 0.6m) but haven't hit the 0.2m absolute danger floor,
                # land immediately if the tag stays missing to avoid drifting blindly.
                elif controller.prev_z < 0.6:
                    print(
                        f"[CRITICAL] Tag lost at low altitude ({controller.prev_z:.2f}m). "
                        "Aborting and forcing immediate touchdown!"
                    )
                    controller.stationary_landing()
                    time.sleep(5)
                    controller.disarm_motors()
                    break

                # --- 3. Standard Search Tiers (Only runs if safely above 0.6m) ---
                elif time_lost < HOVER_TIMEOUT:
                    print(
                        f"[WARN] Tag lost for "
                        f"{time_lost:.1f}s. Coasting on predicted path..."
                    )
                    controller.coast_on_last_velocity(boost_multiplier=COAST_BOOST, vertical_velocity=0.0)

                elif time_lost < SEARCH_TIMEOUT:
                    print(
                        f"[WARN] Tag lost for "
                        f"{time_lost:.1f}s. Ascending to widen FOV..."
                    )
                    controller.coast_on_last_velocity(boost_multiplier=COAST_BOOST, vertical_velocity=-0.2)

                else:
                    print(
                        "[CRITICAL] Tag lost too long high up. "
                        "Emergency blind landing..."
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
                abs(body_x) < 1.0 and
                abs(body_y) < 1.0 and
                body_z < LANDING_THRESHOLD
            ):
                print("[INFO] Landing conditions reached")
                controller.smart_touchdown(timeout=8.0)
                #controller.stationary_landing()
                time.sleep(5)  # Wait for landing to complete
                controller.disarm_motors()
                break

            # track the tag by sending velocity commands
            controller.adjust_velocity_and_send(
                body_x,
                body_y,
                body_z
            )
            # Small sleep to stabilize loop timing
            time.sleep(0.01)