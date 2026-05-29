# This code includes utility functions to request attitude messages for a specified interval continously and get the current attitude message if there is one in the buffer.
# It also has testing where it will request attitude messages and print the values to the console.
from pymavlink import mavutil
from controls.connect import connect_UART0
import time

def get_attitude(master):
    # 1. Check the message buffer for an attitude message and if the buffer does not contain an attitude message return None.
    #    If the message buffer does contain an attitude message, return the roll, pitch, and yaw as a list in radians.
    #    The recv_match function is set to non-blocking so it will not wait for a message. 
    attitude_msg = master.recv_match(type='ATTITUDE', blocking=False)

    if attitude_msg is not None:
        return [attitude_msg.roll, attitude_msg.pitch, attitude_msg.yaw]
    else:
        return None

def request_attitude_messages(master, interval_ms=30):
    print(f"Entered request_attitude_messages() for Target System: {master.target_system} & Target Component: {master.target_component} & Requesting ATTITUDE message stream at {interval_ms}ms intervals")

    # 1. MAVLink expects the interval parameter in microseconds (us) so wo convert the input interval from milliseconds (ms) to microseconds (us) by multiplying by 1000.
    interval_us = interval_ms * 1000

    # 2. We do this because by default the pixhawk does not send certain messages, the ATTITUDE message being one of them.
    #    So we send a command telling the pixhawk to send a specific message at a specific interval.
    #    In the default case we tell it to send the ATTITUDE message every 50ms.
    master.mav.command_long_send(master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
        interval_us, 0, 0, 0, 0, 0)

if __name__ == "__main__":
    master = connect_UART0()
    request_attitude_messages(master, interval_ms=50)
    time.sleep(5)
    print("\nListening for ATTITUDE stream (Non-Blocking)...")
    
    try:
        while True:
            attitude = get_attitude(master)
            
            if attitude is not None:
                roll, pitch, yaw = attitude
                print(f"Roll: {roll:+.4f} | Pitch: {pitch:+.4f} | Yaw: {yaw:+.4f}")
            
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped getting attitude data.")