import numpy as np
import depthai as dai
from multiprocessing import shared_memory

def camera_broadcaster(lock):
    W, H = 640, 400

    shm_rgb = shared_memory.SharedMemory(name="oak_rgb")
    shm_gray = shared_memory.SharedMemory(name="oak_gray")
    shm_depth = shared_memory.SharedMemory(name="oak_depth")

    shared_rgb = np.ndarray((H, W, 3), dtype=np.uint8, buffer=shm_rgb.buf)
    shared_gray = np.ndarray((H, W), dtype=np.uint8, buffer=shm_gray.buf)
    shared_depth = np.ndarray((H, W), dtype=np.uint16, buffer=shm_depth.buf)

    print("[Broadcaster] Shared memory connected. Booting Camera...")

    FPS = 30.0
    with dai.Device() as device:
        with dai.Pipeline(device) as pipeline:
            cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            
            stereo = pipeline.create(dai.node.StereoDepth)
            sync = pipeline.create(dai.node.Sync)

            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(True)
            stereo.setDepthAlign(dai.StereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)

            # NOTE: Added enableUndistortion=True just like your working script
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
            
            print("[Broadcaster] Camera running! Sending frames to RAM...")

            first_frame = True # Flag to help us debug

            while pipeline.isRunning():
                msg_group = sync_q.get()
                if msg_group is None:
                    continue

                rgb_frame = msg_group["rgb"].getCvFrame()
                gray_frame = msg_group["left"].getCvFrame()
                depth_frame = msg_group["depth"].getFrame()

                # Force RGB to 3 channels just in case it brings an alpha channel (BGRA)
                if len(rgb_frame.shape) == 3 and rgb_frame.shape[2] == 4:
                    rgb_frame = rgb_frame[:, :, :3]

                try:
                    # --- CRITICAL SECTION ---
                    with lock:
                        np.copyto(shared_rgb, rgb_frame)
                        np.copyto(shared_gray, gray_frame)
                        np.copyto(shared_depth, depth_frame)
                    
                    if first_frame:
                        print("[Broadcaster] SUCCESS! First frame written to memory.")
                        first_frame = False
                    print("wrote frame")

                except Exception as e:
                    print("\n[CRITICAL BROADCASTER ERROR] The memory copy failed!")
                    print(f"Error Message: {e}")
                    print(f"-> Camera sent RGB shape: {rgb_frame.shape} | Expected: {shared_rgb.shape}")
                    print(f"-> Camera sent Gray shape: {gray_frame.shape} | Expected: {shared_gray.shape}")
                    print(f"-> Camera sent Depth shape: {depth_frame.shape} | Expected: {shared_depth.shape}")
                    print("Shutting down broadcaster...\n")
                    break