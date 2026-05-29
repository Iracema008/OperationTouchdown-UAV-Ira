# This code tests the movement of the drone using custom local positioning, we use the local position messages to calculate our own custom local position 
# and then we use those values to tell the drone to move to specific coordinates relative to its current position. 
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from pymavlink import mavutil


def takeoff(height, uart_tx_mutex):
    print(f"Entered takeoff() for Target System: {master.target_system} & Target Component: {master.target_component}")
    serial_port = '/dev/serial0'
    baudrate =  57600
    source_system = 1
    source_component = 191

    # 1. Takeoff to specified height (positive value for takeoff command) this command using latitude and longitude however if set to 0 
    #    for both then it will ignore them and instead takeoff from current position. This allows the code to work for gps and non-gps implementations.
    with uart_tx_mutex:
        master = mavutil.mavlink_connection(serial_port, baud=baudrate, source_system=source_system, source_component=source_component)
        master.target_system = 1 # Send messages to system 1(drone/vehicle #1)
        master.target_component = 1 # Send messages to flight controller "autopilot"
        master.mav.command_long_send(master.target_system, master.target_component, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,0, 0, 0, 0, 0, 0, 0, height)
        master.close()
    time.sleep(5) # Wait for 5 seconds to allow the drone to takeoff and stabilize at the new height

def land_current_position(uart_tx_mutex):
    print(f"Entered land_current_position() for Target System: {master.target_system} & Target Component: {master.target_component}")
    serial_port = '/dev/serial0'
    baudrate =  57600
    source_system = 1
    source_component = 191

    # 1. Land by using land command, this command using latitude and longitude however if set to 0 for both then it will ignore them
    #    and instead land at current position. This allows the code to work for gps and non-gps implementations. 
    with uart_tx_mutex:
        master = mavutil.mavlink_connection(serial_port, baud=baudrate, source_system=source_system, source_component=source_component)
        master.target_system = 1 # Send messages to system 1(drone/vehicle #1)
        master.target_component = 1 # Send messages to flight controller "autopilot"
        master.mav.command_long_send(master.target_system, master.target_component, mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0)
        master.close()
    time.sleep(5) # Wait for 5 seconds to allow the drone to land and stabilize at the landing position

def move(x, y, z, uart_tx_mutex, position_mutex):
    serial_port = '/dev/serial0'
    baudrate =  57600
    source_system = 1
    source_component = 191
    master = mavutil.mavlink_connection(serial_port, baud=baudrate, source_system=source_system, source_component=source_component)
    print(f"Entered move() for Target System: {master.target_system} & Target Component: {master.target_component}")
    shm_x_coord = shared_memory.SharedMemory(name="pixhawk_x_coord")
    shm_y_coord = shared_memory.SharedMemory(name="pixhawk_y_coord")
    shm_z_coord = shared_memory.SharedMemory(name="pixhawk_z_coord")
    shared_x_coord = np.ndarray((1,), dtype=np.float64, buffer=shm_x_coord.buf)
    shared_y_coord = np.ndarray((1,), dtype=np.float64, buffer=shm_y_coord.buf)
    shared_z_coord = np.ndarray((1,), dtype=np.float64, buffer=shm_z_coord.buf)
    with position_mutex:
        local_x = shared_x_coord[0]
        local_y = shared_y_coord[0]
        local_z = shared_z_coord[0]
    # 1. Tell drone to move to specified position relative to current position using custom origin coordinates reference frame, until within 0.8 meters of target position
    while(abs(local_x - x) > 0.8 or abs(local_y - y) > 0.8 or abs(local_z - z) > 0.8):
        hold_position(x, y, z, 2, uart_tx_mutex)
        with position_mutex:
            local_x = shared_x_coord[0]
            local_y = shared_y_coord[0]
            local_z = shared_z_coord[0]

def goto(x, y, z, uart_tx_mutex):
    serial_port = '/dev/serial0'
    baudrate =  57600
    source_system = 1
    source_component = 191
    # 1. Send SET_POSITION_TARGET_LOCAL_NED message to move the drone to the specified position
    with uart_tx_mutex:
        master = mavutil.mavlink_connection(serial_port, baud=baudrate, source_system=source_system, source_component=source_component)
        master.target_system = 1 # Send messages to system 1(drone/vehicle #1)
        master.target_component = 1 # Send messages to flight controller "autopilot"
        master.mav.set_position_target_local_ned_send(0, master.target_system, master.target_component, mavutil.mavlink.MAV_FRAME_LOCAL_NED, 
            ( mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE),
            x, y, z, # X, Y, Z (NED)
            0, 0, 0, # VX, VY, VZ (not used)
            0, 0, 0, # AX, AY, AZ (not used)
            0, 0) # YAW, YAW_RATE (not used)
        master.close()


def hold_position(x, y, z, duration, uart_tx_mutex):
    # 1. Continously tell drone to go to specified position for specified duration
    start_time = time.time()
    while time.time() - start_time < duration:
        goto(x, y, z, uart_tx_mutex)
        time.sleep(0.05)