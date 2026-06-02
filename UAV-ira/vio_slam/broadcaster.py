'''  Broadcaster process for camera, depth, and Pixhawk telemetry. '''

import numpy as np
import depthai as dai
import time
import cv2

from multiprocessing import shared_memory
from controls.connect import connect_UART2
from controls.attitude import request_attitude_messages, get_attitude
from controls.nedlocalposition import request_local_nedposition_messages, get_local_nedposition
from core.log import get_logger

logger = get_logger(__name__)


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

    logger.info("[BROADCASTER] Shared memory connected")

   # 3. Connect to the Pixhawk via UART2 and request ATTITUDE and LOOCAL_POSITION_NED message streams at the specified intervals
    master_uart2 = connect_UART2()
    request_attitude_messages(master_uart2, 25)
    request_local_nedposition_messages(master_uart2, 40)

    with dai.Device() as device:
        with dai.Pipeline(device) as pipeline:
            logger.info("[BROADCASTER] Starting camera pipeline")

            # creates camera nodes & output queues
            cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

            stereo = pipeline.create(dai.node.StereoDepth)
            sync = pipeline.create(dai.node.Sync)

            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(True)
            stereo.setDepthAlign(
                dai.StereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT
            )

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
            logger.info("[BROADCASTER] Camera running")

            # 5. Get the camera calibration data and write it to the shared memory
            calib = device.getCalibration()
            K = np.array(
                calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, W, H),
                dtype=np.float64
            )
            with depth_frame_mutex:
                np.copyto(shared_calib, K)
            logger.info("[BROADCASTER] Calibration written to shared memory")

            while pipeline.isRunning():
                msg_group = sync_q.get()

                # Read attitude from Pixhawk
                attitude = get_attitude(master_uart2)
                local_position_ned = get_local_nedposition(master_uart2)

                if msg_group is None:
                    continue

                rgb_frame = msg_group["rgb"].getCvFrame()
                gray_frame = msg_group["left"].getCvFrame()
                depth_frame = msg_group["depth"].getFrame()

                # Force RGB to 3 channels
                if len(rgb_frame.shape) == 3 and rgb_frame.shape[2] == 4:
                    rgb_frame = rgb_frame[:, :, :3]

                # Write attitude (pitch, roll, yaw)
                with attitude_mutex:
                    if attitude is not None:
                        shared_attitude[:] = attitude

                # Write local position NED
                with local_position_ned_mutex:
                    if local_position_ned is not None:
                        shared_local_position_ned[:] = local_position_ned

                # Write frames
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
                    logger.error(f"[BROADCASTER] Frame write error: {e}")
                    logger.error(f"  RGB: {rgb_frame.shape} expected {shared_rgb.shape}")
                    logger.error(f"  Gray: {gray_frame.shape} expected {shared_gray.shape}")
                    logger.error(f"  Depth: {depth_frame.shape} expected {shared_depth.shape}")
                    break