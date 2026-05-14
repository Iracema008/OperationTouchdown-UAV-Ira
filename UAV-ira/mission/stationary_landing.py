import time
import depthai as dai
from vision.common.detectors.april_tag_detector import AprilTagDetector
from pixhawk_controller.stationary_landing_controller import StationaryLandingController

# Reminder: Make sure this matches what you found via 'ls /dev/tty*'
CONNECTION_STRING = "/dev/serial0"
BAUDRATE = 57600
LANDING_THRESHOLD = 1.5
TAKEOFF_ALTITUDE = 3 # in meters

# 1. Initialize the Device first
with dai.Device() as device:
    print("[INFO] OAK-D Started")

    # Read calibration before starting the pipeline
    calibration = device.getCalibration()
    
    april_tag_detector = AprilTagDetector(calibration)
    controller = StationaryLandingController(CONNECTION_STRING, BAUDRATE)

    # 2. Create the Pipeline bound to the device
    with dai.Pipeline(device) as pipeline:
        
        # 3. Create and build the Camera Node
        cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        
        # 4. Request Output directly (No XLinkOut needed)
        rgb_out = cam_rgb.requestOutput(
            size=(300, 300), 
            type=dai.ImgFrame.Type.NV12, 
            fps=30
        )

        # 5. Create the queue directly from that output
        q_rgb = rgb_out.createOutputQueue(maxSize=4, blocking=False)

        # 6. Start the pipeline
        pipeline.start()
        print("[INFO] Pipeline started. Initiating takeoff sequence...")

        controller.change_flight_mode("GUIDED")
        controller.arm_motors()
        controller.takeoff_to_altitude(TAKEOFF_ALTITUDE)  # 3 meters

        # --- SETUP: The Fallback Timers ---
        last_tag_time = time.time()
        
        # Extended Fallback thresholds (in seconds)
        HOVER_TIMEOUT = 4.0   # Patient hover duration
        SEARCH_TIMEOUT = 7.0 # Total time before forcing a blind landing

        # 7. Safe loop checking if the pipeline is still active
        while pipeline.isRunning():
            in_rgb = q_rgb.get()
            
            if in_rgb is None:
                continue
                
            frame = in_rgb.getCvFrame()
            pose = april_tag_detector.get_tag_pose(frame)

            # --- LOGIC: The Fallback State Machine ---
            if pose is None:
                time_lost = time.time() - last_tag_time
                
                if time_lost < HOVER_TIMEOUT:
                    print(f"[WARN] Tag lost for {time_lost:.1f}s. Hovering patiently...")
                    controller.send_velocity(0, 0, 0)
                    
                elif time_lost < SEARCH_TIMEOUT:
                    print(f"[WARN] Tag lost for {time_lost:.1f}s. Ascending to widen FOV...")
                    # Ascend slowly at 0.2 m/s
                    controller.send_velocity(0, 0, -0.2) 
                    
                else:
                    print("[CRITICAL] Tag lost for 12+ seconds. Initiating blind VIO landing...")
                    # Descend slowly at 0.3 m/s
                    controller.send_velocity(0, 0, 0.3)
                    
                continue 

            # --- LOGIC: Tag is visible ---
            last_tag_time = time.time() # Reset the safety timer

            cam_x, cam_y, cam_z = pose
            print(f"[INFO] Tag Position (Camera): X={cam_x:.2f}, Y={cam_y:.2f}, Z={cam_z:.2f} m")

            body_x, body_y, body_z = controller.convert_camera_to_body_frame(
                cam_x, cam_y, cam_z
            )

            print(f"[INFO] Tag Position (Body): X={body_x:.2f}, Y={body_y:.2f}, Z={body_z:.2f}")

            if body_z < LANDING_THRESHOLD:
                print("[INFO] Landing threshold reached. Landing...")
                controller.stationary_landing()
                controller.disarm_motors()
                break

            controller.adjust_velocity_and_send(body_x, body_y, body_z)