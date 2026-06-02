''' Main file to run entire UAV, now w/multiprocessing '''

# python main.py --mode scan --planner grid
# python main.py --mode scan --planner sa

import time
import argparse
import multiprocessing as mp
from multiprocessing import Array
from datetime import datetime
from pathlib import Path

from core.state import (
    create_shared_state,
    cleanup_shared_state,
    create_vio_pipeline_state,
    cleanup_vio_pipeline_state,
    UAVStateAccessor,
    FlightMode,
)
from core.config import load_config
from core.log import get_logger

from vio_slam.broadcaster import broadcaster
from vio_slam.vio import run_vio_process
from vio_slam.slam import run_slam_process
from telemetry.telemetry_logger import telemetry_logger
# mission module selected at runtime based on --planner flag

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="UAV autonomous mission")
    parser.add_argument("--mode",    choices=["scan", "land"], default="scan")
    parser.add_argument(
        "--planner", choices=["grid", "sa"], default="grid",
        help="grid = plain lawnmower | sa = simulated annealing"
    )
    args = parser.parse_args()

    cfg = load_config(mode=args.mode)

    # Import the correct mission module based on planner flag
    if args.planner == "sa":
        from mission.mission_sa import run_mission
        logger.info("[MAIN] Planner = SA (simulated annealing)")
    else:
        from mission.mission_grid import run_mission
        logger.info("[MAIN] Planner = GRID (boustrophedon)")

    logger.info(f"[MAIN] Mode={cfg.mode} | Planner={args.planner}")
    logger.info("[MAIN] Creating shared memory")

    # creates ALL shared memory before spawning any process.
    # vio, aruco, flight mode
    state_dict = create_shared_state()
    # rgb,depth, mono
    vio_dict   = create_vio_pipeline_state(
        W=cfg.camera.width,
        H=cfg.camera.height,
    )

    lock = state_dict["lock"]
    marker_confirmed = state_dict["marker_confirmed"]
    ugv_signal = state_dict["ugv_signal"]
    hover_reached = state_dict["hover_reached"]

    # VIO pipeline mutexes, one per shared memory block
    rgb_frame_mutex = vio_dict["rgb_frame_mutex"]
    gray_frame_mutex = vio_dict["gray_frame_mutex"]
    depth_frame_mutex = vio_dict["depth_frame_mutex"]
    attitude_mutex = vio_dict["attitude_mutex"]
    position_mutex = vio_dict["position_mutex"]
    local_position_ned_mutex = vio_dict["local_position_ned_mutex"]
    slam_trigger_mutex = vio_dict["slam_trigger_mutex"]
    slam_enabled_mutex = vio_dict["slam_enabled_mutex"]

    uncertain_pos = Array('d', [0.0, 0.0, 0.0])

    # Main process state accessor for monitoring loop
    state = UAVStateAccessor(lock, marker_confirmed, ugv_signal, hover_reached)

    # Flight log paths
    Path("flight_logs").mkdir(exist_ok=True)
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = f"flight_logs/flight_{log_timestamp}.db"

    logger.info("[MAIN] Shared memory ready — spawning processes")

    # Process 1: Broadcaster, writes camera frames and calibration
    p_broadcaster = mp.Process(
        target=broadcaster,
        args=(
            rgb_frame_mutex,
            gray_frame_mutex,
            depth_frame_mutex,
            attitude_mutex,
            local_position_ned_mutex,
        ),
        name="broadcaster"
    )
    p_broadcaster.start()
    logger.info("[MAIN] broadcaster started — waiting 3s for camera boot")
    # Camera needs ~3s to initialise before VIO can read valid frames
    time.sleep(3)

    # Process 2: VIO, reads from shared memory, computes pose, writes to pose
    p_vio = mp.Process(
        target=run_vio_process,
        args=(
            gray_frame_mutex,
            depth_frame_mutex,
            attitude_mutex,
            position_mutex,
            slam_trigger_mutex,
            lock,
            marker_confirmed,
            ugv_signal,
            hover_reached,
            cfg,
        ),
        name="vio"
    )
    p_vio.start()
    logger.info("[MAIN] vio started — waiting 1s for calibration read")
    # VIO reads calibration from shared memory at startup —
    # give broadcaster time to write it first
    time.sleep(1)


    # Process 3: SLAM, reads RGB from shared memory, finds loop closures, writes drift corrections
    p_slam = mp.Process(
        target=run_slam_process,
        args=(
            rgb_frame_mutex,
            attitude_mutex,
            position_mutex,
            slam_enabled_mutex,
            slam_trigger_mutex,
            cfg,
        ),
        name="slam"
    )
    p_slam.start()
    logger.info("[MAIN] slam started")
    time.sleep(0.5)

    # Process 4: Mission, reads camera, pose from shared memory, runs aruco detection & path planning
    p_mission = mp.Process(
        target=run_mission,
        args=(
            lock,
            marker_confirmed,
            ugv_signal,
            hover_reached,
            cfg,
            log_timestamp,
            uncertain_pos,
            args.planner,
            rgb_frame_mutex,
        ),
        name="mission"
    )
    p_mission.start()
    logger.info("[MAIN] mission started")


    # Process 5: Telemetry, reads UAV state from shared memory, logs to SQLite database.
    p_telemetry = mp.Process(
        target=telemetry_logger,
        args=(
            lock,
            db_path,
            cfg.pixhawk.connection_string,
            cfg.pixhawk.baud_rate,
        ),
        name="telemetry"
    )
    p_telemetry.start()
    logger.info(f"[MAIN] telemetry started — logging to {db_path}")

    processes = [p_broadcaster, p_vio, p_slam, p_mission, p_telemetry]

    logger.info("[MAIN] All 5 processes running")


    # Monitoring loop — runs until mission exits
    try:
        while True:
            mode = state.get_flight_mode()
            (vio_x, vio_y, vio_z, _), vio_ts = state.get_vio_position()
            (_, marker_id) = state.get_aruco_pose()

            vio_str    = (f"{vio_x:.2f},{vio_y:.2f},{vio_z:.2f}"
                          if vio_ts > 0 else "None")
            marker_str = f"ID:{marker_id}" if marker_id else "None"
            sa_str     = (
                f"SA-replan fired at N={uncertain_pos[0]:.2f} "
                f"E={uncertain_pos[1]:.2f}"
                if uncertain_pos[2] == 1.0 else "SA-replan pending"
            )

            logger.info(
                f"[STATUS] mode={mode.name} | "
                f"vio={vio_str} | "
                f"marker={marker_str} | "
                f"{sa_str}"
            )

            # Mission done — sweep complete or marker found and landed
            if not p_mission.is_alive():
                logger.info("[MAIN] Mission finished — shutting down")
                break

            time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info("[MAIN] Keyboard interrupt — shutting down")

    finally:
        logger.info("[MAIN] Terminating all processes")
        for p in processes:
            if p.is_alive():
                p.terminate()

        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                logger.warning(f"[MAIN] Force killing {p.name}")
                p.kill()

        state.close()
        cleanup_shared_state(state_dict)
        cleanup_vio_pipeline_state(vio_dict)

        logger.info(f"[MAIN] Shutdown complete. Telemetry saved to {db_path}")
        logger.info("[MAIN] Done. Yabadabadoo!")


if __name__ == "__main__":
    # spawn is mandatory on Pi, fork breaks DepthAI and OpenCV
    mp.set_start_method('spawn', force=True)
    main()