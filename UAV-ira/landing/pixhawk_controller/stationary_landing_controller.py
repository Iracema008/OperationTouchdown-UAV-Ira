# Stationary Landing Controller
#
# Uses MAVLink to command Pixhawk velocity
# Converts camera frame to body frame
# Implements velocity control
# Lands drone and cuts motors
# Has a takeoff function for testing 
#
# Assumptions:
# Downward facing camera
# Camera centered on drone
# Pixhawk stabilizes drone

from pymavlink import mavutil
import time

# These are proportional control gains (adjust as needed for testing) 
# Controls how aggressively we move to the tag
Kp_xy = 0.5
Kp_z  = 0.4
Ki_xy = 0.1
Kd_xy = 0.25

# Safety limit on velocity commands (adjust as we test)
MAX_VELOCITY = 0.3


class StationaryLandingController:

    def __init__(self, connection_string, baudrate):

        print("[INFO] Connecting to Pixhawk...")

        self.master = mavutil.mavlink_connection(connection_string, baud=baudrate)
        self.master.target_system = 1 # Send messages to system 1(drone/vehicle #1)
        self.master.target_component = 1 # Send messages to flight controller "autopilot"
        print("Waiting for heartbeat...")
        self.master.wait_heartbeat()
        print("Heartbeat Received & Connection Established")
        print(f"Source System: {self.master.source_system}, Source Component: {self.master.source_component}, Target System: {self.master.target_system}, Target Component: {self.master.target_component}, Connection Type: {connection_string}, Baudrate: {baudrate}")
        print("[INFO] Pixhawk Connected")

        self.prev_x = 0
        self.prev_y = 0
        self.prev_z = 0

        self.integral_x = 0.0
        self.integral_y = 0.0
        self.last_vx = 0.0
        self.last_vy = 0.0

        self.last_error_x = 0.0
        self.last_error_y = 0.0
        self.last_time = time.time()
    
    def heartbeat(self):
        print("Waiting for heartbeat from Pixhawk...")
        self.master.wait_heartbeat()
        print(f"Heartbeat from system (system {self.master.target_system} component {self.master.target_component})")
        print(f"Using MAVLink 2.0: {self.master.mavlink20()} \n\n\n")

    def arm_motors(self):
        print(f"Entered arm_drone() & Setting Arming Parameters for Target System: {self.master.target_system} & Target Component: {self.master.target_component}")

        # 1. Setting some arming parameters for the drone, these parameters dicate under what conditions the drone will arm.
        #    These are not all of the parameters however, so if for some reason other parameters are changed then it could fail
        #    to arm. The ARMING_REQUIRE is a parameter for planes, not for drones, this distinction is important as changing
        #    parameters for a plane can result in the drone not arming. So we set ARMING_REQUIRE = 1 which is its default value.
        #    This is to ensure it is always 1 whenever we arm to prevent being unable to arm. We also print out the values it
        #    becomes and the associated parameter. Both are printed to the termal for logging purposes.  
        params = {"ARMING_REQUIRE": 1, "ARMING_CHECK": 1, "ARMING_ACCTHRESH": 0.3, "ARMING_MAGTHRESH": 75, "ARMING_NEED_LOC": 0}
        for name, value in params.items():
            try:
                self.master.mav.param_set_send(self.master.target_system, self.master.target_component, name.encode(), float(value), mavutil.mavlink.MAV_PARAM_TYPE_INT32)
                print(f"Set Parameter ({name}) = {value}")
                msg = self.master.recv_match(type='PARAM_VALUE', blocking=True, timeout=1)
                print(f"MESSAGE: {msg.get_type()}")
                print(f"DATA: {msg.to_dict()}\n")
                if not msg:
                    continue

            except Exception as e:
                print(f"Failed to set {name}: {e}", end=" ")
        self.master.wait_heartbeat()
        print("\nParameters set. You may need to reboot FCU for sensors to reinit.")

        # 2. Here we arm the drone using the built-in helper function from pymavlink/mavutil library
        print("Arming Drone Motors")
        self.master.arducopter_arm()

        # 3. Flush the buffer until we catch the correct command acknowledement(cmd ack) by pulling the next cmd ack from the queue continously.
        #    This is for logging purposes to confirm that the arming command was sent and accepted by the flight controller. It also serves to 
        #    clear the buffer of any old messages until we catch the correct one that shows the drone is armed.
        print("Reading message buffer to catch up to arming change...")
        start_time = time.time()
        while time.time() - start_time < 3: # Continously read command acknowledgements for 3 seconds
            command_ack_msg = self.master.recv_match(type=['COMMAND_ACK'], blocking=True, timeout=2) # Receive a command acknowledgement message and block up to 2 seconds
            if command_ack_msg is not None:
                if command_ack_msg.command == 400 and command_ack_msg.result == 0:
                    print(f"Command Acknowledgment received for MAV_CMD_COMPONENT_ARM_DISARM(CMD #400) with result MAV_RESULT_ACCEPTED(0)")
                    print(f"{command_ack_msg.get_type()}: {command_ack_msg.to_dict()}\n")
                    break
        else:
            if command_ack_msg is not None:
                print(f"Command Acknowledgment received but timed out with wrong command or result: CMD #{command_ack_msg.command} & CMD Result #{command_ack_msg.result}")
                print(f"{command_ack_msg.get_type()}: {command_ack_msg.to_dict()}")
            else:
                print("Timed out waiting for command acknowledgement message")

        # 4. Here we check if the drone is armed using the built-in helper function from pymavlink/mavutil library to ensure that the drone is armed. This is because sometimes 
        #    the drone can fail to arm due to various reasons such as bad parameters, bad GPS lock, or bad sensor readings. So this is a backup to ensure that the drone 
        #    is armed and ready to fly. The previous step is more for logging purposes to confirm that the arming command was sent and accepted by the flight controller, 
        #    but this step is to ensure that the drone is actually armed. And we also print out the result to the terminal for logging purposes.
        print("Checking if drone armed...")
        start_time = time.time()
        while time.time() - start_time < 3: # Wait for up too 3 seconds for confirmation that the motors armed
            self.master.motors_armed_wait()
            if self.master.motors_armed():
                print("Drone is armed and ready to fly\n")
                break
            else:
                print("Escaped motors_armed_wait() but drone did not arm")
        else:
            if self.master.motors_armed():
                print("Timed out waiting for confirmation that drone is armed. But motors show they are armed.\n")
            else:
                print("Motors failed to arm and timed out waiting for confirmation that drone is armed.\n")

    def disarm_motors(self):
        """
        Disarm the drone
        """
        print("Disarming Drone Component(Motors)...\n")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0, # 1 = ARM, 0 = DISARM
            0, 0, 0, 0, 0, 0
        )
        self.master.motors_disarmed_wait()
        print("Motors Disarmed!\n")

    def change_flight_mode(self, flight_mode):
        """
        Change the flight mode of the drone and confirm the change by reading messages from the buffer.
        """
        print(f"Entered change_flight_mode() for Target System: {self.master.target_system} & Target Component: {self.master.target_component}")

        # 1. ArduPilot will actively reject a flight mode switch if its (EKF) hasn't secured a solid GPS lock 
        #    and stabilized its sensors. So this loop waits for 3 seconds to allow for this to happen.
        #    It will also reject a switch/delete newest messeges that the Pixhawk sends to the Pi, this is because
        #    the message buffer overflows. We solve the overflow issue by using the recv_match() helper function from
        #    the pymavlink/mavutil library to read any incoming messages and clear the buffer.
        #    IMPORTANT: SIMPLY DOING TIME.SLEEP() WILL NOT WORK AS WE NEED TO ALSO READ/CLEAR THE BUFFER
        print("Waiting 3 seconds for sensors, GPS, and reading message queue...")
        start_time = time.time()
        while time.time() - start_time < 3:
            self.master.recv_match(blocking=False) # Grab any waiting message and immediately discard it
            time.sleep(0.1)
        self.master.wait_heartbeat()
        print("Done waiting\n")

        # 2. Switch flight mode using the built-in helper function from pymavlink/mavutil library
        print(f"Switching to {flight_mode} flight mode...")
        self.master.set_mode(flight_mode)

        # 3. Flush the buffer until we catch the correct command acknowledement(cmd ack) by pulling the next cmd ack from the queue. 
        #    This is for logging purposes to confirm that the mode change command was sent and accepted by the flight controller. It also 
        #    serves to clear the buffer of any old messages until we catch the correct one that shows the drone is in the specified mode.
        print("Reading message buffer to catch command acknowledgment...")
        start_time = time.time()
        while time.time() - start_time < 3: # Continously read command acknowledgements for 3 seconds
            command_ack_msg = self.master.recv_match(type=['COMMAND_ACK'], blocking=True, timeout=2) # Receive a command acknowledgement message and block up to 2 seconds
            if command_ack_msg is not None:
                if command_ack_msg.command == 176 and command_ack_msg.result == 0:
                    print(f"Command Acknowledgment received for MAV_CMD_DO_SET_MODE(CMD #176) with result MAV_RESULT_ACCEPTED(0)")
                    print(f"{command_ack_msg.get_type()}: {command_ack_msg.to_dict()}\n")
                    break
        else:
            if command_ack_msg is not None:
                print(f"Command Acknowledgment received but timed out with wrong command or result: CMD #{command_ack_msg.command} & CMD Result #{command_ack_msg.result}")
                print(f"{command_ack_msg.get_type()}: {command_ack_msg.to_dict()}")
            else:
                print("Timed out waiting for command acknowledgement message")

        # 4. Flush the buffer until we catch the updated heartbeat by pulling the next heartbeat from the queue. Pymavlink caches the flight mode based
        #    on the LAST heartbeat it read. If we print the mode immediately, it will falsely print the wrong mode. To solve this we must actively pull 
        #    new heartbeats from the queue until we catch up to the message that proves the flight controller actually did switch modes.
        print("Reading message buffer to get latest heartbeat...")
        start_time = time.time()
        while time.time() - start_time < 3: # Continously read heartbeats for 3 seconds
            heartbeat_msg = self.master.recv_match(type=['HEARTBEAT'], blocking=True, timeout=2) # Recieve a heartbeat message and block up to 2 second
            if (heartbeat_msg is not None) and (self.master.flightmode == flight_mode):
                print(f"Succesfully switched to: {self.master.flightmode} flight mode & Base Mode(MAV_MODE_FLAGS): {self.master.base_mode}\n")
                break
        else:
            print(f"Timed out waiting for {flight_mode}. Current mode seen: {self.master.flightmode} & Base Mode(MAV_MODE_FLAGS): {self.master.base_mode}\n")

    def stationary_landing(self):
        """
        Send land command to Pixhawk
        """
        print("[INFO] Landing...")

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0,
            0,0,0,0,0,0,0
        )
        time.sleep(5)

    def disable_safety_checks(self):
        """
        Disable safety and arming checks for bench tests.
        """
        print("Disabling safety switch and arming checks...")
        params = {"ARMING_REQUIRE": 1, "ARMING_CHECK": 1, "ARMING_ACCTHRESH": 0.255, "ARMING_MAGTHRESH": 50, "ARMING_NEED_LOC": 0}
        for name, value in params.items():
            try:
                self.master.mav.param_set_send(self.master.target_system, self.master.target_component,
                                        name.encode(), float(value),
                                        mavutil.mavlink.MAV_PARAM_TYPE_INT32)
                print(f"Set Parameter ({name}) = {value}")
                msg = self.master.recv_match(type='PARAM_VALUE', blocking=True, timeout=1)
                print(f"MESSAGE: {msg.get_type()}")
                print(f"DATA: {msg.to_dict()}")

                time.sleep(0.2)
            except Exception as e:
                print(f"Failed to set {name}: {e}")

        print("\nParameters sent. You may need to reboot FCU for sensors to reinit.")

    def takeoff_to_altitude(self, meters):
        """
        Take off to specified altitude (meters)
        """
        print(f"[INFO] Taking off to {meters} meters...")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0,0,0,0,0,0,
            meters
        )
        time.sleep(5)

    def send_velocity(self, vx, vy, vz):
        """
        Send velocity command to Pixhawk
        """
        # Send the velocity command to the Pixhawk using MAVLink
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000111111000111,
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0, 0
        )

    def convert_camera_to_body_frame(self, cam_x, cam_y, cam_z):
        """
        Convert camera frame to the body frame
        """
        # Camera frame to body frame conversion
        #
        # Downward camera: (Got this from chat, we need to check this with actual testing))
        # body_x = -cam_y
        # body_y =  cam_x
        # body_z =  cam_z

        body_x = -cam_y
        body_y = cam_x
        body_z = cam_z

        return body_x, body_y, body_z
    
    def smart_touchdown(self, timeout=8.0):
        """
        Send a steady downward velocity and monitor Pixhawk telemetry to 
        detect when physical downward movement stops (touchdown).
        """
        print("[INFO] Initiating smart descent to touchdown...")
        start_time = time.time()
        first_stopped_time = None
        
        # Flush the buffer of old VFR_HUD messages so we don't read past telemetry
        while self.master.recv_match(type='VFR_HUD', blocking=False):
            pass
            
        # Push down until we detect the floor OR the timeout is reached
        while time.time() - start_time < timeout:
            # Command 0.2 m/s straight down
            self.send_velocity(0.0, 0.0, 0.2) 
            
            # Pull the latest HUD telemetry to check actual climb rate
            msg = self.master.recv_match(type='VFR_HUD', blocking=True, timeout=0.1)
            
            if msg:
                # msg.climb is the vertical speed in m/s.
                # If it is near 0 (e.g., < 0.1 m/s), the drone is physically blocked by the floor.
                # We wait 1 second before checking to give the drone time to build downward momentum first.
                if abs(msg.climb) < 0.1 and (time.time() - start_time > 1.0):
                    if first_stopped_time is None:
                        first_stopped_time = time.time()
                        
                    # If it has been physically stopped for 0.5 consecutive seconds, it is landed
                    if time.time() - first_stopped_time >= 0.5:
                        print(f"[INFO] Touchdown detected! Vertical speed flatlined at {msg.climb:.2f} m/s.")
                        break
                else:
                    # Drone is still moving down, reset the stopped timer
                    first_stopped_time = None
            else:
                # If no message caught, just sleep briefly to stabilize
                time.sleep(0.05)
                
        print("[INFO] Touchdown sequence finished.")

    def coast_on_last_velocity(self, boost_multiplier=1.2, vertical_velocity=0.0):
        """
        Keep moving in the last known direction with a boost multiplier 
        to ensure we don't fall behind a moving target while blind.
        """
        boosted_vx = self.last_vx * boost_multiplier
        boosted_vy = self.last_vy * boost_multiplier
        
        # We can also handle the vertical velocity here for the search phase
        self.send_velocity(boosted_vx, boosted_vy, vertical_velocity)

    def adjust_velocity_and_send(self, body_x, body_y, body_z):
        """
        Apply full PID (Proportional-Integral-Derivative) control, 
        gain scheduling, anti-ground effect, and send velocity command.
        """
        # --- 1. Signal Smoothing ---
        alpha = 0.7
        self.prev_x = alpha*self.prev_x + (1-alpha)*body_x
        self.prev_y = alpha*self.prev_y + (1-alpha)*body_y
        self.prev_z = alpha*self.prev_z + (1-alpha)*body_z

        body_x = self.prev_x
        body_y = self.prev_y
        body_z = self.prev_z

        # --- 2. Predictive Tracking (Derivative Control) ---
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: 
            dt = 0.01  # Prevent division by zero

        derivative_x = (body_x - self.last_error_x) / dt
        derivative_y = (body_y - self.last_error_y) / dt

        self.last_error_x = body_x
        self.last_error_y = body_y
        self.last_time = current_time

        # --- 3. Gain Scheduling & Dynamic Thresholding ---
        if body_z < 0.5:
            thresh = 0.10  
            current_kp_xy = Kp_xy * 0.4  
            current_max_vel = MAX_VELOCITY * 0.5 
        elif body_z < 2.0:
            thresh = 0.10  
            current_kp_xy = Kp_xy * 0.7
            current_max_vel = MAX_VELOCITY * 0.8
        else:
            thresh = 0.05  
            current_kp_xy = Kp_xy
            current_max_vel = MAX_VELOCITY

        # Apply deadband
        body_x = 0 if abs(body_x) < thresh else body_x
        body_y = 0 if abs(body_y) < thresh else body_y

        # --- 4. Speed Matching (Integral Control) ---
        if abs(body_x) > 0:
            self.integral_x += body_x * dt
        else:
            self.integral_x *= 0.9  
            
        if abs(body_y) > 0:
            self.integral_y += body_y * dt
        else:
            self.integral_y *= 0.9

        # Anti-windup cap
        max_integral = 0.2
        self.integral_x = max(min(self.integral_x, max_integral), -max_integral)
        self.integral_y = max(min(self.integral_y, max_integral), -max_integral)

        # --- 5. Vertical Target ---
        TARGET_Z = 0.0  # Kept at 0.0 so error remains high all the way to the floor
        error_z = body_z - TARGET_Z

        # --- 6. Final Velocity Math (P + I + D) ---
        vx = (current_kp_xy * body_x) + (Ki_xy * self.integral_x) + (Kd_xy * derivative_x)
        vy = (current_kp_xy * body_y) + (Ki_xy * self.integral_y) + (Kd_xy * derivative_y)
        vz = 0 if abs(error_z) < 0.05 else Kp_z * error_z

        # --- 7. Anti-Ground Effect Override ---
        # If attempting to descend, force a minimum downward speed of 0.15 m/s
        if vz > 0:
            vz = max(vz, 0.15)

        # --- 8. Safety Clipping ---
        vx = max(min(vx, current_max_vel), -current_max_vel)
        vy = max(min(vy, current_max_vel), -current_max_vel)
        vz = max(min(vz, current_max_vel), -current_max_vel)

        # --- 9. Save State & Execute ---
        self.last_vx = vx
        self.last_vy = vy

        self.send_velocity(vx, vy, vz)