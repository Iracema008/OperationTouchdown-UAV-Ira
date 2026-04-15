import multiprocessing as mp
from multiprocessing import shared_memory

from broadcaster import camera_broadcaster
from viewer import run_viewer

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    W, H = 640, 400
    RGB_BYTES = W * H * 3
    GRAY_BYTES = W * H
    DEPTH_BYTES = W * H * 2  # 16-bit depth uses 2 bytes per pixel

    print("[Main] Allocating Shared Memory in RAM...")
    
    # 1. Create all THREE shared memory blocks
    shm_rgb = shared_memory.SharedMemory(create=True, size=RGB_BYTES, name="oak_rgb")
    shm_gray = shared_memory.SharedMemory(create=True, size=GRAY_BYTES, name="oak_gray")
    shm_depth = shared_memory.SharedMemory(create=True, size=DEPTH_BYTES, name="oak_depth")
    
    # 2. Create the Mutex Lock
    frame_lock = mp.Lock()

    # 3. Define the independent processes
    broadcaster_process = mp.Process(target=camera_broadcaster, args=(frame_lock,))
    viewer_process = mp.Process(target=run_viewer, args=(frame_lock,))

    try:
        # 4. Start the processes
        broadcaster_process.start()
        viewer_process.start()

        # Wait until you close the viewer window
        viewer_process.join()
        
        # Gracefully kill the broadcaster
        broadcaster_process.terminate()
        broadcaster_process.join()

    except KeyboardInterrupt:
        print("\n[Main] Caught Keyboard Interrupt. Shutting down...")
    finally:
        # 5. Cleanup all THREE blocks
        print("[Main] Cleaning up Shared Memory...")
        shm_rgb.close()
        shm_rgb.unlink()
        shm_gray.close()
        shm_gray.unlink()
        shm_depth.close()
        shm_depth.unlink()
        print("[Main] All processes terminated safely.")