"""
Telemetry Logger Process

Runs as an independent process alongside camera/VIO/vision.
Polls Pixhawk via MAVLink for battery, attitude, mode, and system status.
Reads VIO position from shared memory.
Writes everything to SQLite with a unified timeline.

Usage (from main.py):
    import multiprocessing as mp
    from telemetry.telemetry_logger import telemetry_logger
    
    logger_proc = mp.Process(
        target=telemetry_logger,
        args=(frame_lock, "/path/to/flight_log.db", "/dev/serial0", 921600)
    )
    logger_proc.start()
"""

import time
import sqlite3
import numpy as np
from datetime import datetime
from multiprocessing import shared_memory

try:
    from pymavlink import mavutil
except ImportError:
    print("[TELEMETRY] ERROR: pymavlink not installed")
    print("Install: pip install pymavlink pyserial")
    raise


def _init_database(db_path: str):
    """Create SQLite schema if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")  # Process-safe writes
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wall_time_iso TEXT NOT NULL,
            t_sec REAL NOT NULL,
            
            -- VIO position (from shared memory)
            vio_x_m REAL,
            vio_y_m REAL,
            vio_z_m REAL,
            vio_yaw_deg REAL,
            
            -- Pixhawk attitude
            roll_deg REAL,
            pitch_deg REAL,
            yaw_deg REAL,
            
            -- Battery
            battery_voltage_v REAL,
            battery_current_a REAL,
            battery_remaining_pct INTEGER,
            
            -- Flight state
            flight_mode TEXT,
            armed INTEGER,
            ekf_ok INTEGER,
            
            -- System health
            cpu_load_pct INTEGER,
            voltage_drop_mah INTEGER
        )
    """)
    
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_time ON telemetry(t_sec)
    """)
    
    conn.commit()
    conn.close()
    print(f"[TELEMETRY] Database initialized: {db_path}")


def telemetry_logger(lock, db_path: str, mavlink_device: str = "/dev/serial0", baud: int = 921600):
    """
    Main telemetry logging loop.
    
    Args:
        lock: multiprocessing.Lock for shared memory access
        db_path: path to SQLite database file
        mavlink_device: serial port for Pixhawk connection
        baud: baud rate (default 921600 for /dev/serial0)
    """
    print(f"[TELEMETRY] Starting logger process")
    print(f"[TELEMETRY] DB: {db_path}")
    print(f"[TELEMETRY] MAVLink: {mavlink_device} @ {baud}")
    
    # Initialize database
    _init_database(db_path)
    
    # Connect to Pixhawk
    print("[TELEMETRY] Connecting to Pixhawk...")
    try:
        master = mavutil.mavlink_connection(mavlink_device, baud=baud)
        master.wait_heartbeat(timeout=10)
        print(f"[TELEMETRY] Heartbeat OK (system {master.target_system})")
    except Exception as e:
        print(f"[TELEMETRY] FATAL: Failed to connect to Pixhawk: {e}")
        return
    
    # Connect to VIO shared memory
    try:
        shm_vio = shared_memory.SharedMemory(name="oak_vio")
        shared_vio = np.ndarray((4,), dtype=np.float64, buffer=shm_vio.buf)
        print("[TELEMETRY] Connected to VIO shared memory")
    except FileNotFoundError:
        print("[TELEMETRY] WARNING: VIO shared memory not found, position logging disabled")
        shm_vio = None
        shared_vio = None
    
    # State tracking
    t0 = time.time()
    last_attitude = None
    last_battery = None
    last_heartbeat = None
    last_sys_status = None
    
    LOG_RATE_HZ = 10.0
    log_period = 1.0 / LOG_RATE_HZ
    last_log_time = 0.0
    
    conn = sqlite3.connect(db_path)
    
    print(f"[TELEMETRY] Logger running at {LOG_RATE_HZ} Hz")
    
    try:
        while True:
            now = time.time()
            t_sec = now - t0
            
            # Poll MAVLink messages (non-blocking)
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
            
            # Log at fixed rate
            if (now - last_log_time) < log_period:
                time.sleep(0.001)
                continue
            
            last_log_time = now
            wall_iso = datetime.now().isoformat(timespec="milliseconds")
            
            # Read VIO position from shared memory
            vio_x, vio_y, vio_z, vio_yaw = None, None, None, None
            if shared_vio is not None:
                with lock:
                    local_vio = shared_vio.copy()
                vio_x, vio_y, vio_z, vio_yaw = local_vio
            
            # Extract MAVLink data
            roll_deg = pitch_deg = yaw_deg = None
            if last_attitude:
                roll_deg = np.degrees(last_attitude.roll)
                pitch_deg = np.degrees(last_attitude.pitch)
                yaw_deg = np.degrees(last_attitude.yaw)
            
            battery_v = battery_a = battery_pct = None
            if last_battery:
                # BATTERY_STATUS reports in millivolts / milliamps
                battery_v = last_battery.voltages[0] / 1000.0 if last_battery.voltages[0] != 65535 else None
                battery_a = last_battery.current_battery / 100.0 if last_battery.current_battery != -1 else None
                battery_pct = last_battery.battery_remaining if last_battery.battery_remaining != -1 else None
            
            flight_mode = None
            armed = None
            if last_heartbeat:
                flight_mode = master.flightmode
                armed = 1 if (last_heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) else 0
            
            ekf_ok = None
            cpu_load = None
            voltage_drop = None
            if last_sys_status:
                # EKF health bit is in onboard_control_sensors_health
                ekf_flags = last_sys_status.onboard_control_sensors_health
                ekf_ok = 1 if (ekf_flags & mavutil.mavlink.MAV_SYS_STATUS_AHRS) else 0
                cpu_load = last_sys_status.load / 10  # Reported in 0.1% units
                voltage_drop = last_sys_status.voltage_battery - last_sys_status.voltage_battery
            
            # Write to database
            conn.execute("""
                INSERT INTO telemetry (
                    wall_time_iso, t_sec,
                    vio_x_m, vio_y_m, vio_z_m, vio_yaw_deg,
                    roll_deg, pitch_deg, yaw_deg,
                    battery_voltage_v, battery_current_a, battery_remaining_pct,
                    flight_mode, armed, ekf_ok,
                    cpu_load_pct, voltage_drop_mah
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                wall_iso, t_sec,
                vio_x, vio_y, vio_z, vio_yaw,
                roll_deg, pitch_deg, yaw_deg,
                battery_v, battery_a, battery_pct,
                flight_mode, armed, ekf_ok,
                cpu_load, voltage_drop
            ))
            
            # Commit every 2 seconds
            if int(t_sec) % 2 == 0:
                conn.commit()
    
    except KeyboardInterrupt:
        print("\n[TELEMETRY] Keyboard interrupt, shutting down")
    
    finally:
        conn.commit()
        conn.close()
        if shm_vio:
            shm_vio.close()
        print("[TELEMETRY] Logger stopped, database closed")
