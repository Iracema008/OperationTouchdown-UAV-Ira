''' Software In The Loop (SITL) runner for lawnmower search & landing. '''

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

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    # sa  → mission/mission_sa.py  (plain distance SA)
    # sac → mission/mission_sa_center.py  (center-biased SA)
    # grid → mission/mission_grid.py  (boustrophedon)
    parser.add_argument(
        "--planner", choices=["grid", "sa", "sac"], default="grid"
    )
    args = parser.parse_args()

    # 1. Dynamic import — same pattern as main.py
    if args.planner == "sa":
        from mission.mission_sa import run_mission
        logger.info("[SITL] Planner = SA (plain distance)")
    elif args.planner == "sac":
        from mission.mission_sa_center import run_mission
        logger.info("[SITL] Planner = SAC (center-biased)")
    else:
        from mission.mission_grid import run_mission
        logger.info("[SITL] Planner = GRID (boustrophedon)")

    cfg = load_config(mode="scan")

    # 2. UAV state shared memory only — no VIO pipeline blocks needed for SITL
    state_dict       = create_shared_state()
    lock             = state_dict["lock"]
    marker_confirmed = state_dict["marker_confirmed"]
    ugv_signal       = state_dict["ugv_signal"]
    hover_reached    = state_dict["hover_reached"]

    uncertain_pos   = Array('d', [0.0, 0.0, 0.0])

    # No OAK-D in SITL — fake mutex so mission process doesn't crash on import
    rgb_frame_mutex = mp.Lock()

    Path("flight_logs").mkdir(exist_ok=True)
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    state = UAVStateAccessor(lock, marker_confirmed, ugv_signal, hover_reached)

    # 3. Keyboard trigger thread — Press Enter at any point to simulate
    #    ArUco marker confirmed. Drone will stop sweeping and transition
    #    to approach → land at its current position.
    #    In SA/SAC missions this also triggers the destination_discovered
    #    log event and approach phase exactly as it would on real hardware.
    def wait_for_keypress():
        input("\n>>> Press Enter to simulate marker_confirmed <<<\n")
        if not marker_confirmed.is_set():
            marker_confirmed.set()
            logger.info(
                "[SITL] marker_confirmed fired — "
                "drone will abort search and descend"
            )

    threading.Thread(target=wait_for_keypress, daemon=True).start()

    # 4. Single SITL mission process — reads LOCAL_POSITION_NED from SITL
    #    directly on UART0 (udp:0.0.0.0:14550 in config).
    #    All other processes (broadcaster, vio, slam) are skipped.
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

    # 5. Monitoring loop — prints position and confirmed status every second
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