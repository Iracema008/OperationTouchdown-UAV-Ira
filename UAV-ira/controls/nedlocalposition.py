# This code includes utility functions to request local position ned messages for a specified interval continously and get the current local position
# ned message if there is one in the buffer. It also has testing where it will request local position ned messages and print the values to the console.
from pymavlink import mavutil
from controls.connect import connect_UART0
import time

def get_local_nedposition(master):
    # 1. Check the message buffer for an local position ned message and if the buffer does not contain an local position ned message return None.
    #    If the message buffer does contain an local position ned message, return a list containing the north east and down position in that order
    #    The recv_match function is set to non-blocking so it will not wait for a message.
    local_nedposition_msg = master.recv_match(type=['LOCAL_POSITION_NED'], blocking=False)

    if local_nedposition_msg is not None:
        return [local_nedposition_msg.x, local_nedposition_msg.y, local_nedposition_msg.z]
    else:
        return None

def request_local_nedposition_messages(master, interval_ms=50):
    print(f"Entered request_local_nedposition_messages() for Target System: {master.target_system} & Target Component: {master.target_component} & Requesting LOCAL_POSITION_NED message stream at {interval_ms}ms intervals")

    # 1. MAVLink expects the interval parameter in microseconds (us) so wo convert the input interval from milliseconds (ms) to microseconds (us) by multiplying by 1000.
    interval_us = interval_ms * 1000

    # 2. We do this because by default pixhawk does not send certain messages, the LOCAL_POSITION_NED message being one of them.
    #    So we send a command telling the pixhawk to send a message at a certain interval, in the default case we tell it to send the 
    #    LOCAL_POSITION_NED message every 50ms.
    master.mav.command_long_send(master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0, mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED,
        interval_us, 0, 0, 0, 0, 0)

if __name__ == "__main__":
    master = connect_UART0()
    request_local_nedposition_messages(master, interval_ms=50)
    time.sleep(5)
    print("\nListening for LOCAL_POSITION_NED stream (Non-Blocking)...")
    
    try:
        while True:
            local_position_ned = get_local_nedposition(master)
            
            if local_position_ned is not None:
                north, east, down = local_position_ned
                print(f"North: {north:+.4f} | East: {east:+.4f} | Down: {down:+.4f}")
            
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped getting local position data.")