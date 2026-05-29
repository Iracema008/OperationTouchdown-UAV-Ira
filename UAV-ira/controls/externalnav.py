# This code includes a utility function that changes parameters such that the drone uses External Navigation as its main positioning system.
# The testing code when run as main simply runs the utility function. The parameters are for ardupilot copter parameters.
from pymavlink import mavutil
from controls.connect import connect_UART0
#from TestComponents.test_change_flight_mode import change_flight_mode

def externalnav(master):
    print(f"Entered externalnav() & Setting External Navigation Parameters for Target System: {master.target_system} & Target Component: {master.target_component}")

    # 1. Setting External Navigation parameters for the drone, these parameters dicate what the drone will use for External Navigation and positioning.
    #    These are not all of the parameters however, so if for some reason other parameters are changed then it could fail
    #    to arm or work properly. We also print out the values it becomes and the associated parameter. Both are printed to the terminal for logging purposes.  
    params = {"EK2_ENABLE": 0, "EK3_ENABLE": 1, "AHRS_EKF_TYPE": 3, "GPS1_TYPE": 0, "GPS_PRIMARY": 0, 
              "EK3_SRC1_POSXY": 6, "EK3_SRC1_VELXY": 0, "EK3_SRC1_POSZ": 6, "EK3_SRC1_VELZ": 0, "EK3_SRC1_YAW": 1}
    for name, value in params.items():
        try:
            master.mav.param_set_send(master.target_system, master.target_component, name.encode(), float(value), mavutil.mavlink.MAV_PARAM_TYPE_INT32)
            print(f"Set Parameter ({name}) = {value}")
            msg = master.recv_match(type='PARAM_VALUE', blocking=True, timeout=1)
            print(f"MESSAGE: {msg.get_type()}")
            print(f"DATA: {msg.to_dict()}\n")
            if not msg:
                continue

        except Exception as e:
            print(f"Failed to set {name}: {e}", end=" ")
    master.wait_heartbeat()
    print("\nParameters set. You may need to reboot FCU for sensors to reinit.")

if __name__ == "__main__":
    master = connect_UART0()
    #change_flight_mode(master, "GUIDED")
    externalnav(master)