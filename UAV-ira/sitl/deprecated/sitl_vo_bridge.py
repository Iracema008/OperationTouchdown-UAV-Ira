import time
from pymavlink import mavutil
from core.state import UAVStateAccessor
from core.log import get_logger

logger = get_logger(__name__)


def run_sitl_bridge(lock, marker_confirmed, ugv_signal, hover_reached, cfg):
    logger.info("[SITL BRIDGE] Starting")
    state = UAVStateAccessor(lock, marker_confirmed, ugv_signal, hover_reached)

    master = None
    for attempt in range(10):
        try:
            logger.info(f"[SITL BRIDGE] Connection attempt {attempt + 1}/10")
            master = mavutil.mavlink_connection(
                "tcp:127.0.0.1:5760",
                baud=57600,
                source_system=1,
                source_component=192   # mission uses 191, bridge uses 192
            )
            master.wait_heartbeat(timeout=5)
            master.target_system    = 1
            master.target_component = 1
            logger.info("[SITL BRIDGE] Connected — heartbeat received")
            break
        except Exception as e:
            logger.warning(f"[SITL BRIDGE] Attempt {attempt + 1} failed: {e}")
            master = None
            time.sleep(2)

    if master is None:
        logger.error("[SITL BRIDGE] Could not connect after 10 attempts")
        state.close()
        return

    master.mav.request_data_stream_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_POSITION,
        10,    # 10Hz
        1      # start sending
    )
    logger.info("[SITL BRIDGE] Requested POSITION stream at 10Hz")
    # temporary debug — add after the request, replace the while loop
    for _ in range(20):
        msg = master.recv_match(blocking=True, timeout=1.0)
        if msg:
            logger.info(f"[SITL BRIDGE] Got msg type: {msg.get_type()}")
            
    #logger.info("[SITL BRIDGE] Requested LOCAL_POSITION_NED at 10Hz")

    try:
        while True:
            msg = master.recv_match(
                type='LOCAL_POSITION_NED',
                blocking=True,
                timeout=1.0
            )
            if msg is not None:
                state.set_vio_position(
                    float(msg.x),   # north
                    float(msg.y),   # east
                    float(msg.z),   # down
                    0.0
                )
                logger.debug(
                    f"[SITL BRIDGE] N={msg.x:.2f} "
                    f"E={msg.y:.2f} D={msg.z:.2f}"
                )
            else:
                logger.debug("[SITL BRIDGE] Waiting for position msg...")

    except KeyboardInterrupt:
        logger.info("[SITL BRIDGE] Interrupted")
    finally:
        state.close()
        logger.info("[SITL BRIDGE] Exiting")