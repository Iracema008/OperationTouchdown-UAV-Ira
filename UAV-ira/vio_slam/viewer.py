import cv2
import numpy as np
import time
from multiprocessing import shared_memory

def run_viewer(lock):
    """
    Reads frames from shared memory and displays them side-by-side.
    """
    W, H = 640, 400
    time.sleep(5)
    # We only connect to the two blocks we actually care about drawing
    shm_rgb = shared_memory.SharedMemory(name="oak_rgb")
    shm_gray = shared_memory.SharedMemory(name="oak_gray")

    shared_rgb = np.ndarray((H, W, 3), dtype=np.uint8, buffer=shm_rgb.buf)
    shared_gray = np.ndarray((H, W), dtype=np.uint8, buffer=shm_gray.buf)

    local_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    local_gray = np.zeros((H, W), dtype=np.uint8)

    print("[Viewer] Connected to Shared Memory. Starting display...")

    while True:
        # --- CRITICAL SECTION ---
        with lock:
            np.copyto(local_rgb, shared_rgb)
            np.copyto(local_gray, shared_gray)
        # ------------------------

        # Make the 1-channel gray into 3-channels so we can glue them together
        gray_bgr = cv2.cvtColor(local_gray, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((local_rgb, gray_bgr))

        cv2.imshow("Parallel Viewer (RGB | Gray)", combined)
        print("showed frame")
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break

    print("[Viewer] Exiting...")
    cv2.destroyAllWindows()
    
    shm_rgb.close()
    shm_gray.close()