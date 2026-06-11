# This code establishes a connection to the pixhawk and returns a mavlink master object, depending on which function is used it will connect to a different UART port on the Pi. 
# Note that only one connection can be established to a UART port at a time, so if you try to connect to the same UART port twice without closing the first connection then the 
# second connection attempt will fail. A port may only establish a connection with a single process at a time. 
from pymavlink import mavutil

def connect_UART0():
    serial_port = '/dev/ttyS3'
    baudrate =  57600
    source_system = 1 # System/vehicle sending messages is system1(drone/vehicle #1)
    source_component = 191 # Component sending messages is onboard Computer

    print("\nConnecting to Pixhawk via UART0 & waiting for heartbeat")
    master = mavutil.mavlink_connection(serial_port, baud=baudrate, source_system=source_system, source_component=source_component)
    master.target_system = 1 # Send messages to system 1(drone/vehicle #1)
    master.target_component = 1 # Send messages to flight controller "autopilot"
    master.wait_heartbeat()
    print("Heartbeat received & connection established for UART0")
    print(f"Source System: {master.source_system}, Source Component: {master.source_component}, Target System: {master.target_system}, Target Component: {master.target_component}, Connection Type: {serial_port}, Baudrate: {baudrate}")
    
    return master

def connect_UART2():
    serial_port = '/dev/ttyS4'
    baudrate =  57600
    source_system = 1 # System/vehicle sending messages is system1(drone/vehicle #1)
    source_component = 191 # Component sending messages is onboard Computer

    print("\nConnecting to Pixhawk via UART2 & waiting for heartbeat")
    master = mavutil.mavlink_connection(serial_port, baud=baudrate, source_system=source_system, source_component=source_component)
    master.target_system = 1 # Send messages to system 1(drone/vehicle #1)
    master.target_component = 1 # Send messages to flight controller "autopilot"
    master.wait_heartbeat()
    print("Heartbeat received & connection established for UART2")
    print(f"Source System: {master.source_system}, Source Component: {master.source_component}, Target System: {master.target_system}, Target Component: {master.target_component}, Connection Type: {serial_port}, Baudrate: {baudrate}")
    
    return master

def connect_UART3():
    serial_port = '/dev/ttyS7'
    baudrate =  57600
    source_system = 1 # System/vehicle sending messages is system1(drone/vehicle #1)
    source_component = 191 # Component sending messages is onboard Computer

    print("\nConnecting to Pixhawk via UART3 & waiting for heartbeat")
    master = mavutil.mavlink_connection(serial_port, baud=baudrate, source_system=source_system, source_component=source_component)
    master.target_system = 1 # Send messages to system 1(drone/vehicle #1)
    master.target_component = 1 # Send messages to flight controller "autopilot"
    master.wait_heartbeat()
    print("Heartbeat received & connection established for UART3")
    print(f"Source System: {master.source_system}, Source Component: {master.source_component}, Target System: {master.target_system}, Target Component: {master.target_component}, Connection Type: {serial_port}, Baudrate: {baudrate}")
    
    return master

if __name__ == "__main__":
    master0 = connect_UART0()
    master2 = connect_UART2()
    master3 = connect_UART3()

    master0.close()
    master2.close()
    master3.close()