""" Software In The Loop (SITL) runner for lawnmower search & landing."""

import time
import threading
import argparse
import multiprocessing as mp
from multiprocessing import Array
from datetime import datetime
from pathlib import Path


from core.state import (
    create_shared_state,
    cleanup_shared_state,
    UAVStateAccessor,
)

from core.config import load_config
from core.log import get_logger
#from mission.mission import run_mission

logger = get_logger(__name__)


# running with grid is lawnmower, simulated annealing is sa
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", choices=["grid", "sa"], default="grid")
    args = parser.parse_args()

    # Dynamic import — same pattern as main.py
    if args.planner == "sa":
        from mission.mission_sa import run_mission
        logger.info("[SITL] Planner = SA")
    else:
        from mission.mission_grid import run_mission
        logger.info("[SITL] Planner = GRID")

    cfg = load_config(mode="scan")

    # UAV state shared memory only
    # no VIO pipeline blocks needed for sitl
    state_dict       = create_shared_state()
    lock             = state_dict["lock"]
    marker_confirmed = state_dict["marker_confirmed"]
    ugv_signal       = state_dict["ugv_signal"]
    hover_reached    = state_dict["hover_reached"]

    uncertain_pos   = Array('d', [0.0, 0.0, 0.0])

    # no actual camera is needed for sitl, so we keep a fake mutex
    rgb_frame_mutex = mp.Lock()

    Path("flight_logs").mkdir(exist_ok=True)
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    state = UAVStateAccessor(lock, marker_confirmed, ugv_signal, hover_reached)


    # Press Enter to simulate marker found at any point during searcg
    def wait_for_keypress():
        input("\n>>> Press Enter to simulate marker_confirmed <<<\n")
        if not marker_confirmed.is_set():
            marker_confirmed.set()
            logger.info(
                "[SITL] marker_confirmed fired — "
                "drone will abort search and descend"
            )

    threading.Thread(target=wait_for_keypress, daemon=True).start()

    # single SITL connection, local pos ned from SITL inside its own loop
    p_mission = mp.Process(
        target=run_mission,
        args=(
            lock, marker_confirmed, ugv_signal, hover_reached,
            cfg, log_timestamp, uncertain_pos,
            args.planner, rgb_frame_mutex,
        ),
        name="mission"
    )
    p_mission.start()
    logger.info("[SITL] Mission process started")

    try:
        while True:
            (x, y, z, _), ts = state.get_vio_position()
            vio_str = f"{x:.2f},{y:.2f},{z:.2f}" if ts > 0 else "None"
            logger.info(
                f"[STATUS] vio={vio_str} | "
                f"confirmed={marker_confirmed.is_set()}"
            )

            if not p_mission.is_alive():
                logger.info("[SITL] Mission finished")
                break

            time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info("[SITL] Keyboard interrupt")

    finally:
        if p_mission.is_alive():
            p_mission.terminate()
        p_mission.join(timeout=5)
        if p_mission.is_alive():
            p_mission.kill()
        state.close()
        cleanup_shared_state(state_dict)
        logger.info("[SITL] Done. Yabadabadoo!")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()