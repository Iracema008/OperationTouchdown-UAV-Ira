""" Real-time health monitoring alert, watches for low battery, EKF fails, mod changes, arm/disarms """

import time
from pymavlink import mavutil


def health_monitor(mavlink_device: str = "/dev/serial0", baud: int = 921600):
    # a separate process from telemetry_logger so the alerts don't block logging if something else fails
  
    print("[HEALTH] Starting monitor")
    
    master = mavutil.mavlink_connection(mavlink_device, baud=baud)
    master.wait_heartbeat(timeout=10)
    
    last_mode = None
    last_armed = None
    last_battery_warning = 0
    
    BATTERY_WARN_THRESHOLD_V = 10.5
    BATTERY_CRITICAL_THRESHOLD_V = 10.0
    
    while True:
        msg = master.recv_match(blocking=False)
        if not msg:
            time.sleep(0.01)
            continue
        
        msg_type = msg.get_type()
        
        if msg_type == "HEARTBEAT":
            current_mode = master.flightmode
            armed = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
            
            if current_mode != last_mode:
                print(f"\033[93m[HEALTH] Mode change: {last_mode} → {current_mode}\033[0m")
                last_mode = current_mode
            
            if armed != last_armed:
                if armed:
                    print(f"\033[92m[HEALTH] ✓ ARMED\033[0m")
                else:
                    print(f"\033[91m[HEALTH] ✗ DISARMED\033[0m")
                last_armed = armed
        
        elif msg_type == "BATTERY_STATUS":
            voltage = msg.voltages[0] / 1000.0 if msg.voltages[0] != 65535 else None
            
            if voltage and (time.time() - last_battery_warning) > 5.0:
                if voltage < BATTERY_CRITICAL_THRESHOLD_V:
                    print(f"\033[91m[HEALTH] ⚠ CRITICAL BATTERY: {voltage:.2f}V\033[0m")
                    last_battery_warning = time.time()
                elif voltage < BATTERY_WARN_THRESHOLD_V:
                    print(f"\033[93m[HEALTH] ⚠ Low battery: {voltage:.2f}V\033[0m")
                    last_battery_warning = time.time()
        
        elif msg_type == "SYS_STATUS":
            ekf_flags = msg.onboard_control_sensors_health
            ekf_ok = (ekf_flags & mavutil.mavlink.MAV_SYS_STATUS_AHRS) != 0
            
            if not ekf_ok:
                print(f"\033[91m[HEALTH] ✗ EKF FAILURE\033[0m")


if __name__ == "__main__":
    health_monitor()
