import numpy as np
import depthai as dai
import time
import cv2
import multiprocessing as mp
from multiprocessing import shared_memory
from controls.connect import connect_UART2
from controls.attitude import request_attitude_messages
from controls.attitude import get_attitude
from controls.nedlocalposition import request_local_nedposition_messages
from controls.nedlocalposition import get_local_nedposition

def broadcaster(rgb_frame_mutex, gray_frame_mutex, depth_frame_mutex, attitude_mutex, local_position_ned_mutex):
    W, H = 640, 400
    FPS = 30.0
    # 1. Connect to the shared memory for RGB, gray, depth, calibration, attitude, and local position NED
    shm_rgb = shared_memory.SharedMemory(name="oak_rgb")
    shm_gray = shared_memory.SharedMemory(name="oak_gray")
    shm_depth = shared_memory.SharedMemory(name="oak_depth")
    shm_calib = shared_memory.SharedMemory(name="oak_calib")
    shm_attitude = shared_memory.SharedMemory(name="attitude")
    shm_local_position_ned = shared_memory.SharedMemory(name="local_position_ned")
    # 2. Create numpy arrays that use the shared memory buffers for RGB, gray, depth, calibration, attitude, and local position NED
    shared_rgb = np.ndarray((H, W, 3), dtype=np.uint8, buffer=shm_rgb.buf)
    shared_gray = np.ndarray((H, W), dtype=np.uint8, buffer=shm_gray.buf)
    shared_depth = np.ndarray((H, W), dtype=np.uint16, buffer=shm_depth.buf)
    shared_calib = np.ndarray((3, 3), dtype=np.float64, buffer=shm_calib.buf)
    shared_attitude = np.ndarray((3,), dtype=np.float64, buffer=shm_attitude.buf)
    shared_local_position_ned = np.ndarray((3,), dtype=np.float64, buffer=shm_local_position_ned.buf)
    print("Broadcaster shared memory connected")
    # 3. Connect to the Pixhawk via UART2 and request ATTITUDE and LOOCAL_POSITION_NED message streams at the specified intervals
    master_uart2 = connect_UART2()
    request_attitude_messages(master_uart2, 25)
    request_local_nedposition_messages(master_uart2, 40)

    with dai.Device() as device:
        with dai.Pipeline(device) as pipeline:
            print("Starting up camera")
            # 4. Create the three camera nodes for RGB, left gray, and right gray, and the stereo depth node.
            #    As well as the sync node to synchronize the frames from all three cameras together. Then create
            #    the output queues for the synchronized frames using the camera nodes and the stereo depth node 
            #    as the sources. And lastly start the camera frame pipeline.
            cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            
            stereo = pipeline.create(dai.node.StereoDepth)
            sync = pipeline.create(dai.node.Sync)

            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(True)
            stereo.setDepthAlign(dai.StereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)

            rgb_out = cam_rgb.requestOutput(size=(W, H), fps=FPS, enableUndistortion=True)
            left_out = cam_left.requestOutput(size=(W, H), fps=FPS)
            right_out = cam_right.requestOutput(size=(W, H), fps=FPS)

            left_out.link(stereo.left)
            right_out.link(stereo.right)
            
            rgb_out.link(sync.inputs["rgb"])
            stereo.rectifiedLeft.link(sync.inputs["left"])
            stereo.depth.link(sync.inputs["depth"])

            sync_q = sync.out.createOutputQueue(maxSize=4, blocking=False)
            pipeline.start()
            print("Camera running")
            # 5. Get the camera calibration data and write it to the shared memory
            calib = device.getCalibration()
            K = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, W, H), dtype=np.float64)
            with depth_frame_mutex:
                np.copyto(shared_calib, K)
            print("Camera calibration data written to shared memory.")
            print("Broadcaster entering main loop to get camera frames and attitude data")
            
            while pipeline.isRunning():
                msg_group = sync_q.get()
                # 6. Get the attitude data from the Pixhawk and store it in the local attitude variable. 
                attitude = get_attitude(master_uart2)
                # 7. Get the local position ned data from the Pixhawk and store it in the local local position ned variable.
                local_position_ned = get_local_nedposition(master_uart2)
                if msg_group is None:
                    continue
                # 8. Get the RGB, gray, and depth frames from the camera and store them in the the local frame variables.
                rgb_frame = msg_group["rgb"].getCvFrame()
                gray_frame = msg_group["left"].getCvFrame()
                depth_frame = msg_group["depth"].getFrame()
                # 9. Force RGB to 3 channels just in case it brings an alpha channel (BGRA)
                if len(rgb_frame.shape) == 3 and rgb_frame.shape[2] == 4:
                    rgb_frame = rgb_frame[:, :, :3]
                # 10. This section is a critical section as we must aquire the mutex/lock before 
                #     writing to the shared memory, and release it immediately after.
                with attitude_mutex:
                    if attitude is not None:
                        shared_attitude[:] = attitude
                # 11. This section is a critical section as we must aquire the mutex/lock before 
                #     writing to the shared memory, and release it immediately after.
                with local_position_ned_mutex:
                    if local_position_ned is not None:
                        shared_local_position_ned[:] = local_position_ned
                try:
                    # 12. This section is a critical section as we must aquire the mutex/lock before 
                    #    copying from the shared memory, and release it immediately after.
                    with gray_frame_mutex:
                        np.copyto(shared_gray, gray_frame)
                    with depth_frame_mutex:
                        np.copyto(shared_depth, depth_frame)
                    with rgb_frame_mutex:
                        np.copyto(shared_rgb, rgb_frame)
                except Exception as e:
                    print("\nBroadcaster error, memory write failed")
                    print(f"Error Message: {e}")
                    print(f"Camera sent RGB shape: {rgb_frame.shape} | Expected: {shared_rgb.shape}")
                    print(f"Camera sent Gray shape: {gray_frame.shape} | Expected: {shared_gray.shape}")
                    print(f"Camera sent Depth shape: {depth_frame.shape} | Expected: {shared_depth.shape}")
                    print("Shutting down broadcaster...\n")
                    break

def test_viewer(camera_frame_mutex, attitude_mutex, local_position_ned_mutex):
    W, H = 640, 400
    time.sleep(4) # Give the broadcaster a moment to allocate memory and boot the camera

    # 1. Connect to the shared memory blocks
    shm_rgb = shared_memory.SharedMemory(name="oak_rgb")
    shm_gray = shared_memory.SharedMemory(name="oak_gray")
    shm_attitude = shared_memory.SharedMemory(name="attitude")
    shm_local_position_ned = shared_memory.SharedMemory(name="local_position_ned")

    # 2. Create the NumPy arrays backed by the shared memory
    shared_rgb = np.ndarray((H, W, 3), dtype=np.uint8, buffer=shm_rgb.buf)
    shared_gray = np.ndarray((H, W), dtype=np.uint8, buffer=shm_gray.buf)
    shared_attitude = np.ndarray((3,), dtype=np.float64, buffer=shm_attitude.buf)
    shared_local_position_ned = np.ndarray((3,), dtype=np.float64, buffer=shm_local_position_ned.buf)

    # 3. Create local variables to hold the copies
    local_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    local_gray = np.zeros((H, W), dtype=np.uint8)
    local_attitude = np.zeros((3,), dtype=np.float64)
    local_position_ned = np.zeros((3,), dtype=np.float64)

    print("Viewer connected to Shared Memory. Starting display...")

    while True:
        # 4. Acquire the mutex/lock before copying from the shared memory, and release it immediately after. This is a critical section.
        with camera_frame_mutex:
            np.copyto(local_rgb, shared_rgb)
            np.copyto(local_gray, shared_gray)
            
        # 5. Acquire the mutex/lock before copying the attitude data from shared memory, and release it immediately after. This is a critical section.
        with attitude_mutex:
            np.copyto(local_attitude, shared_attitude)
        # 6. Acquire the mutex/lock before copying the attitude data from shared memory, and release it immediately after. This is a critical section.
        with local_position_ned_mutex:
            np.copyto(local_position_ned, shared_local_position_ned)

        # 7. Print the attitude data to the terminal
        print(f"Attitude | Roll: {local_attitude[0]:+.4f} | Pitch: {local_attitude[1]:+.4f} | Yaw: {local_attitude[2]:+.4f}")
        # 8. Print the local position data to the terminal
        print(f"Local Position NED | North: {local_position_ned[0]:+.4f} | East: {local_position_ned[1]:+.4f} | Down: {local_position_ned[2]:+.4f}")

        # 9. Combine and display the images
        gray_bgr = cv2.cvtColor(local_gray, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((local_rgb, gray_bgr))
        cv2.imshow("RGB and Gray", combined)
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break
    
    print("Viewer exiting...")
    cv2.destroyAllWindows()
    shm_rgb.close()
    shm_gray.close()
    shm_attitude.close()
    shm_local_position_ned.close()

if __name__ == "__main__":
    # 1. Set the multiprocessing start method to 'spawn' this is so that when a new process is started it
    #    doesn't inherit the memory of the parent process, and instead creates its own memory space and also
    #    its own python interpreter. Effectively isolating the processes from each other except for the shared memory.
    mp.set_start_method('spawn', force=True)
    W, H = 640, 400
    RGB_BYTES = W * H * 3
    GRAY_BYTES = W * H
    DEPTH_BYTES = W * H * 2 
    CALIB_BYTES = 3 * 3 * 8 
    ATTITUDE_BYTES = 3 * 8 # 3 float64 numbers (Roll, Pitch, Yaw = 24 bytes)
    LOCAL_POSITION_NED_BYTES = 3 * 8 # 3 float64 numbers (North, East, Down = 24 bytes)

    # 2. Create the shared memory for RGB, gray, depth, camera calibration matrix, and attitude
    print("Broadcaster tester allocating shared memory...")
    shm_rgb = shared_memory.SharedMemory(create=True, size=RGB_BYTES, name="oak_rgb")
    shm_gray = shared_memory.SharedMemory(create=True, size=GRAY_BYTES, name="oak_gray")
    shm_depth = shared_memory.SharedMemory(create=True, size=DEPTH_BYTES, name="oak_depth")
    shm_calib = shared_memory.SharedMemory(create=True, size=CALIB_BYTES, name="oak_calib")
    shm_attitude = shared_memory.SharedMemory(create=True, size=ATTITUDE_BYTES, name="attitude")
    shm_local_position_ned = shared_memory.SharedMemory(create=True, size=LOCAL_POSITION_NED_BYTES, name="local_position_ned")
    
    # 3. Initialize the three required locks
    camera_frame_mutex = mp.Lock()
    attitude_mutex = mp.Lock()
    local_position_ned_mutex = mp.Lock()

    # 4. Define the processes
    broadcaster_process = mp.Process(target=broadcaster, args=(camera_frame_mutex, attitude_mutex, local_position_ned_mutex))
    viewer_process = mp.Process(target=test_viewer, args=(camera_frame_mutex, attitude_mutex, local_position_ned_mutex))

    try:
        # 5. Start the broadcaster and viewer processes
        broadcaster_process.start()
        viewer_process.start()

        # 6. Wait until the user presses 'q' to close the viewer window
        viewer_process.join()
        
        # 7. Kill the broadcaster once the viewer is closed
        broadcaster_process.terminate()
        broadcaster_process.join()
        
    except KeyboardInterrupt:
        print("\nBroadcaster tester caught keyboard interrupt. Shutting down...")
    finally:
        # 8. Clean up the shared memory(no memory leaks)
        print("Broadcaster tester cleaning up shared memory...")
        shm_rgb.close()
        shm_rgb.unlink()
        shm_gray.close()
        shm_gray.unlink()
        shm_depth.close()
        shm_depth.unlink()
        shm_calib.close()
        shm_calib.unlink()
        shm_attitude.close()
        shm_attitude.unlink()
        shm_local_position_ned.close()
        shm_local_position_ned.unlink()
        print("Broadcaster tester processes terminated safely.")