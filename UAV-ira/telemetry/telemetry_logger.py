''' Telemetry CSV Logger, outputs Appendix B requirments for qual test '''

import csv
import time
import math
import numpy as np

from datetime import datetime
from multiprocessing import shared_memory
from pymavlink import mavutil
from core.log import get_logger

logger = get_logger(__name__)

LOG_RATE_HZ = 10.0
LOG_PERIOD  = 1.0 / LOG_RATE_HZ

CSV_COLUMNS = [
    "wall_time", "t_sec",
    "vio_north_m", "vio_east_m", "vio_down_m", "vio_yaw_deg",
    "vel_north_ms", "vel_east_ms", "vel_down_ms", "speed_ms",
    "roll_deg", "pitch_deg", "yaw_deg",
    "battery_v", "battery_pct",
    "flight_mode", "armed", "ekf_ok",
    "event",
]


def _round(val, digits=3):
    '''Round for cleaner CSV output, return empty string for None.'''
    if val is None:
        return ""
    return round(val, digits)


def telemetry_csv(lock, csv_path: str, mavlink_device: str, baud: int):
    '''connects to Pixhawk, reads VIO pose, writes one row per 100ms to CSV  '''
    print(f"[TELEMETRY] Starting CSV logger")
    print(f"[TELEMETRY] CSV: {csv_path}")
    print(f"[TELEMETRY] MAVLink: {mavlink_device} @ {baud}")

    # 1. Connect to Pixhawk
    try:
        master = mavutil.mavlink_connection(mavlink_device, baud=baud)
        master.wait_heartbeat(timeout=10)
        print("[TELEMETRY] Heartbeat OK")
    except Exception as e:
        print(f"[TELEMETRY] Failed to connect: {e}")
        return

    # 2. Connect to VIO shared memory (uav_vio written by vio process)
    try:
        shm_vio    = shared_memory.SharedMemory(name="uav_vio")
        shared_vio = np.ndarray((5,), dtype=np.float64, buffer=shm_vio.buf)
        print("[TELEMETRY] VIO shared memory connected")
    except Exception:
        print("[TELEMETRY] VIO shared memory not found — position logging disabled")
        shm_vio    = None
        shared_vio = None

    t0= time.time()
    last_log_time = 0.0
    last_attitude = None
    last_battery= None
    last_heartbeat = None
    last_sys_status = None
    last_local_pos = None

    # 3. Open CSV and write header
    csvfile = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)

    writer.writeheader()
    csvfile.flush()
    print(f"[TELEMETRY] Logging at {LOG_RATE_HZ} Hz")

    try:
        while True:
            now   = time.time()
            t_sec = now - t0

            # 4. Poll MAVLink non-blocking — grab whatever message is available
            msg = master.recv_match(blocking=False)
            if msg:
                msg_type = msg.get_type()
                if msg_type == "ATTITUDE":
                    last_attitude = msg
                elif msg_type == "BATTERY_STATUS":
                    last_battery = msg
                elif msg_type == "HEARTBEAT":
                    last_heartbeat = msg
                elif msg_type == "SYS_STATUS":
                    last_sys_status = msg
                elif msg_type == "LOCAL_POSITION_NED":
                    # velocity comes from here — vx, vy, vz in m/s
                    last_local_pos = msg

            if (now - last_log_time) < LOG_PERIOD:
                time.sleep(0.001)
                continue

            last_log_time = now
            wall_time     = datetime.now().isoformat(timespec="milliseconds")

            # 5. Read VIO position from shared memory
            vio_north = vio_east = vio_down = vio_yaw = None
            if shared_vio is not None:
                with lock:
                    local_vio = shared_vio.copy()

                vio_north = local_vio[0]
                vio_east = local_vio[1]
                vio_down = local_vio[2]
                vio_yaw = math.degrees(local_vio[3])

            # 6. Velocity from LOCAL_POSITION_NED
            #    speed_ms is horizontal speed only (north + east components)
            #    this is what the qual test wants for velocity display
            vel_north = vel_east = vel_down = speed = None
            if last_local_pos:
                vel_north = last_local_pos.vx
                vel_east = last_local_pos.vy
                vel_down = last_local_pos.vz
                speed = math.sqrt(vel_north**2 + vel_east**2)

            # 7. Attitude
            roll_deg = pitch_deg = yaw_deg = None
            if last_attitude:
                roll_deg = math.degrees(last_attitude.roll)
                pitch_deg = math.degrees(last_attitude.pitch)
                yaw_deg = math.degrees(last_attitude.yaw)

            # 8. Battery
            battery_v = battery_pct = None
            if last_battery:
                if last_battery.voltages[0] != 65535:
                    battery_v = last_battery.voltages[0] / 1000.0
                if last_battery.battery_remaining != -1:
                    battery_pct = last_battery.battery_remaining

            # 9. Flight mode and arming state
            flight_mode = armed = ekf_ok = None
            if last_heartbeat:
                flight_mode = master.flightmode
                armed = 1 if (
                    last_heartbeat.base_mode &
                    mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                ) else 0
            if last_sys_status:
                ekf_ok = 1 if (
                    last_sys_status.onboard_control_sensors_health &
                    mavutil.mavlink.MAV_SYS_STATUS_AHRS
                ) else 0

            writer.writerow({
                "wall_time":    wall_time,
                "t_sec": _round(t_sec),
                "vio_north_m": _round(vio_north),
                "vio_east_m": _round(vio_east),
                "vio_down_m": _round(vio_down),
                "vio_yaw_deg": _round(vio_yaw, 1),
                "vel_north_ms": _round(vel_north),
                "vel_east_ms": _round(vel_east),
                "vel_down_ms": _round(vel_down),
                "speed_ms": _round(speed),
                "roll_deg": _round(roll_deg, 1),
                "pitch_deg": _round(pitch_deg, 1),
                "yaw_deg": _round(yaw_deg, 1),
                "battery_v": _round(battery_v, 2),
                "battery_pct": battery_pct,
                "flight_mode": flight_mode,
                "armed": armed,
                "ekf_ok": ekf_ok,
                "event": "",
            })

            # Flush every 2 seconds so data survives a crash mid-flight
            if int(t_sec) % 2 == 0:
                csvfile.flush()

    except KeyboardInterrupt:
        print("[TELEMETRY] Interrupted")

    finally:
        csvfile.flush()
        csvfile.close()
        if shm_vio:
            shm_vio.close()
        print(f"[TELEMETRY] CSV saved to {csv_path}")


def log_event(csv_path: str, event: str, extra: dict = None):
    ''' Write a single event row to the CSV from within the mission process,
        doesnt need the telemetry process to be running, writes directly to the same CSV file.'''
    wall_time = datetime.now().isoformat(timespec="milliseconds")
    event_str = event

    if extra:
        parts = " ".join(f"{k}={v}" for k, v in extra.items())
        event_str = f"{event} {parts}"
    try:
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            row = {col: "" for col in CSV_COLUMNS}
            row["wall_time"] = wall_time
            row["t_sec"] = round(time.time(), 3)
            row["event"] = event_str
            writer.writerow(row)
    except Exception as e:
        logger.error(f"[TELEMETRY] Failed to write event '{event}': {e}")