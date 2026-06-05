''' Main file to run entire UAV, now w/multiprocessing '''

### just for testing search :
# python main.py --mode scan --planner grid
# python main.py --mode scan --planner sa

### for challenge 1 and 2
# python main.py --mode scan --planner c1
# python main.py --mode scan --planner c2

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
# mission module is selected at runtime based on --planner flag
# telemetry is merged into mission via log_event() in telemetry/telemetry_csv.py

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="UAV autonomous mission")
    parser.add_argument("--mode", choices=["scan", "land"], default="scan")
    parser.add_argument(
        "--planner", choices=["grid", "sa", "c1", "c2"], default="grid",
        help="grid = boustrophedon | sa = simulated annealing | c1 = challenge 1 | c2 = challenge 2"
    )
    args = parser.parse_args()

    cfg = load_config(mode=args.mode)

    # 1. Import the correct mission module based on --planner flag.
    #    This is done at runtime so we don't import all three every time.
    if args.planner == "sa":
        from mission.mission_sa import run_mission
        logger.info("[MAIN] Planner = SA (simulated annealing)")
    elif args.planner == "c1":
        from mission.challenge_one import run_mission
        logger.info("[MAIN] Planner = C1 (challenge 1 — UGV landing)")
    elif args.planner == "c2":
        from mission.challenge_two import run_mission
        logger.info("[MAIN] Planner = C2 (challenge 2 — ArUco search + UGV landing)")
    else:
        from mission.mission_grid import run_mission
        logger.info("[MAIN] Planner = GRID (boustrophedon)")

    logger.info(f"[MAIN] Mode={cfg.mode} | Planner={args.planner}")
    logger.info("[MAIN] Creating shared memory")

    # 2. Create ALL shared memory before spawning any process.
    #    Child processes connect to existing blocks — they never create them.
    #    state_dict — UAV state (uav_vio, uav_aruco, flight mode etc.)
    #    vio_dict   — VIO pipeline (oak_rgb, oak_gray, oak_depth etc.)
    state_dict = create_shared_state()
    vio_dict   = create_vio_pipeline_state(
        W=cfg.camera.width,
        H=cfg.camera.height,
    )

    lock             = state_dict["lock"]
    marker_confirmed = state_dict["marker_confirmed"]
    ugv_signal       = state_dict["ugv_signal"]
    hover_reached    = state_dict["hover_reached"]

    # One mutex per shared memory block — each process acquires the lock
    # before reading or writing so frames are never half-written
    rgb_frame_mutex          = vio_dict["rgb_frame_mutex"]
    gray_frame_mutex         = vio_dict["gray_frame_mutex"]
    depth_frame_mutex        = vio_dict["depth_frame_mutex"]
    attitude_mutex           = vio_dict["attitude_mutex"]
    position_mutex           = vio_dict["position_mutex"]
    local_position_ned_mutex = vio_dict["local_position_ned_mutex"]
    slam_trigger_mutex       = vio_dict["slam_trigger_mutex"]
    slam_enabled_mutex       = vio_dict["slam_enabled_mutex"]

    # Shared between mission (writes on uncertain detection) and
    # mission SA planner (reads to reorder remaining waypoints)
    # Layout: Array('d', [north, east, flag])
    uncertain_pos = Array('d', [0.0, 0.0, 0.0])

    # Main process state accessor for the monitoring loop
    state = UAVStateAccessor(lock, marker_confirmed, ugv_signal, hover_reached)

    Path("flight_logs").mkdir(exist_ok=True)
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("[MAIN] Shared memory ready — spawning processes")

    # 3. Process 1: Broadcaster — sole OAK-D owner, writes camera frames,
    #    calibration, and attitude to shared memory. Must start first so VIO
    #    and mission have valid frames to read.
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
    logger.info("[MAIN] broadcaster started — waiting 5s for camera boot")
    time.sleep(5)   # camera needs ~3s to initialise before VIO reads frames

    # 4. Process 2: VIO — reads gray and depth from shared memory, computes
    #    NED position via optical flow + PnP, sends vision_position_estimate
    #    to Pixhawk via UART3, writes position to uav_vio for mission to read.
    #    ArUco detection is NOT here — that is mission's responsibility.
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
    logger.info("[MAIN] vio started — waiting 3s for calibration read")
    time.sleep(3)   # VIO reads calibration from shared memory at startup

    # 5. Process 3: SLAM — reads RGB from shared memory, finds loop closures,
    #    writes drift corrections for VIO to apply on the next frame.
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
    time.sleep(1)

    # 6. Process 4: Mission — reads RGB and position from shared memory,
    #    runs ArUco or AprilTag detection, executes path planning and landing.
    #    Sole UART0 owner for arm → sweep → land.
    #    Telemetry is merged here — log_event() writes to CSV from mission process.
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
    logger.info(f"[MAIN] Flight log → flight_logs/flight_{log_timestamp}.csv")

    processes = [p_broadcaster, p_vio, p_slam, p_mission]
    logger.info("[MAIN] All 4 processes running")

    # 7. Monitoring loop — prints status every second until mission exits.
    #    Mission exiting is the signal to shut everything else down.
    try:
        while True:
            mode = state.get_flight_mode()
            (vio_x, vio_y, vio_z, _), vio_ts = state.get_vio_position()
            (_, marker_id) = state.get_aruco_pose()

            vio_str    = (f"{vio_x:.2f},{vio_y:.2f},{vio_z:.2f}" if vio_ts > 0 else "None")
            marker_str = f"ID:{marker_id}" if marker_id else "None"

            logger.info(
                f"[STATUS] mode={mode.name} | "
                f"vio={vio_str} | "
                f"marker={marker_str}"
            )

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
        logger.info("[MAIN] Shutdown complete")
        logger.info("[MAIN] Done. Yabadabadoo!")


if __name__ == "__main__":
    # spawn is mandatory on Pi — fork breaks DepthAI and OpenCV
    mp.set_start_method('spawn', force=True)
    main()