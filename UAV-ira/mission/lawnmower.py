"""
lawnmower_mission.py

Lawnmower search pattern controller that integrates with vo_full_v3.py.

- Reads position from your existing VO_LK instance via vo.pose()
- Sends MAVLink SET_POSITION_TARGET_LOCAL_NED via the existing pymavlink connection
- Geofence defined in LOCAL meters from takeoff origin (0, 0, 0)
- Runs in a background thread so VO loop continues uninterrupted

Coordinate note:
  Your VO frame:  x=right, y=down, z=forward
  MAVLink NED:    x=North, y=East, z=Down (negative = up)
  We work in a simple 2D local frame (forward/right) and let
  MavlinkVisionPublisher handle the NED rotation via yaw_offset.
  For mission commands we use MAV_FRAME_LOCAL_NED directly.

Usage:
  from lawnmower_mission import LawnmowerMission
  mission = LawnmowerMission(vo=vo, mav_master=vision_pub.master)
  mission.start()
"""

import math
import time
import threading
import numpy as np
from pymavlink import mavutil

### make sure your drone is facing North at launch for the simplified frame to hold — or rotate the waypoints by your initial compass heading if not.
MISSION_CONFIG = {
    "flight_alt_m":   2.0,    # altitude AGL to fly the pattern
    "takeoff_alt_m":  2.0,    # takeoff altitude before mission starts

    # Search box in local frame from takeoff origin
    # forward = +Z in camera frame / +X in NED (approximately)
    # right   = +X in camera frame / +Y in NED (approximately)
    "forward_min_m":  0.0,
    "forward_max_m":  8.0,    # total depth of search area
    "right_min_m":    0.0,
    "right_max_m":    8.0,    # total width of search area

    "lane_spacing_m": 1.0,    # distance between parallel passes

    # Navigation tuning
    "wp_accept_radius_m": 0.35,  # how close = "arrived at waypoint"
    "wp_timeout_s":       30.0,  # max time to reach a waypoint before skipping
    "geofence_margin_m":  0.3,   # soft fence triggers return this far inside boundary

    # Safety
    "max_velocity_ms":    1.5,   # clamp for velocity override mode
    "geofence_return_alt_m": 2.0,
}


def _ned_from_vo_frame(forward_m: float, right_m: float) -> tuple:
    """
    Convert (forward, right) in VO/body frame to approximate NED.
    This is a simplified mapping assuming the drone launches facing North.
    For full yaw compensation, rotate by heading at launch.
    NED: x=North=forward, y=East=right, z=Down
    """
    return forward_m, right_m  # (north, east) — adjust if launch heading differs


def _dist2d(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)




# Waypoint generator
def generate_lawnmower_waypoints(cfg: dict) -> list:
    """
    Returns list of (north_m, east_m, down_m) waypoints.
    Sweeps forward, steps right, alternates direction.
    down_m is negative (altitude above ground in NED).
    """
    waypoints = []
    alt_down = -cfg["flight_alt_m"]  # NED z is negative-up

    right = cfg["right_min_m"]
    direction = 1  # +1 = increasing forward, -1 = decreasing

    while right <= cfg["right_max_m"] + 1e-6:
        north_near, east_near = _ned_from_vo_frame(
            cfg["forward_min_m"] if direction == 1 else cfg["forward_max_m"],
            right
        )
        north_far, east_far = _ned_from_vo_frame(
            cfg["forward_max_m"] if direction == 1 else cfg["forward_min_m"],
            right
        )

        waypoints.append((north_near, east_near, alt_down))
        waypoints.append((north_far,  east_far,  alt_down))

        right += cfg["lane_spacing_m"]
        direction *= -1

    return waypoints




# Geofence checker
class LocalGeofence:
    def __init__(self, cfg: dict):
        self.fwd_min = cfg["forward_min_m"] - cfg["geofence_margin_m"]
        self.fwd_max = cfg["forward_max_m"] + cfg["geofence_margin_m"]
        self.rgt_min = cfg["right_min_m"]   - cfg["geofence_margin_m"]
        self.rgt_max = cfg["right_max_m"]   + cfg["geofence_margin_m"]

    def inside(self, north_m: float, east_m: float) -> bool:
        # In our simplified frame: north≈forward, east≈right
        return (self.fwd_min <= north_m <= self.fwd_max and
                self.rgt_min <= east_m  <= self.rgt_max)


class MavCommander:
    def __init__(self, master):
        self.master = master

    def set_mode(self, mode: str):
        mode_id = self.master.mode_mapping()[mode]
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )
        print(f"[Mission] Mode → {mode}")

    def arm(self):
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )
        print("[Mission] Arm sent — waiting...")
        self.master.motors_armed_wait()
        print("[Mission] Armed ✓")

    def takeoff(self, alt_m: float):
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, alt_m
        )
        print(f"[Mission] Takeoff to {alt_m}m sent")
        time.sleep(alt_m / 0.8 + 2.0)  # rough wait: ~0.8 m/s climb + buffer

    def land(self):
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        print("[Mission] LAND sent")

    def goto_ned(self, north: float, east: float, down: float):
        """
        Send position setpoint in LOCAL_NED frame.
        down is negative-up (e.g. -2.0 = 2m above ground).
        """
        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            north, east, down,
            0, 0, 0,
            0, 0, 0,
            0, 0
        )

    def return_to_origin(self, alt_m: float):
        print("[Mission] GEOFENCE BREACH — returning to origin")
        self.goto_ned(0.0, 0.0, -alt_m)




# Mission state machine
class LawnmowerMission:
    """
    Runs the lawnmower pattern in a background thread.

    Parameters
    ----------
    vo          : VO_LK instance from vo_full_v3.py
    mav_master  : the pymavlink connection from MavlinkVisionPublisher.master
    cfg         : optional override dict for MISSION_CONFIG
    auto_arm    : if True, arm+takeoff automatically; set False for bench testing
    """

    STATES = ["IDLE", "ARMING", "TAKEOFF", "FLYING", "GEOFENCE_RTL", "LANDING", "DONE"]

    def __init__(self, vo, mav_master, cfg: dict = None, auto_arm: bool = True):
        self.vo = vo
        self.cfg = {**MISSION_CONFIG, **(cfg or {})}
        self.cmd = MavCommander(mav_master)
        self.fence = LocalGeofence(self.cfg)
        self.waypoints = generate_lawnmower_waypoints(self.cfg)
        self.auto_arm = auto_arm

        self._state = "IDLE"
        self._thread = None
        self._stop_evt = threading.Event()

        self._current_wp_idx = 0
        self._wp_start_time = None

        print(f"[Mission] {len(self.waypoints)} waypoints generated")
        for i, wp in enumerate(self.waypoints):
            print(f"  WP{i:02d}: N={wp[0]:.1f}  E={wp[1]:.1f}  D={wp[2]:.1f}")

    # public API
    def start(self):
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="LawnmowerMission"
        )
        self._thread.start()
        print("[Mission] Thread started")

    def stop(self):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        print("[Mission] Stopped")

    @property
    def state(self):
        return self._state

    # internal state machine 
    def _run(self):
        try:
            if self.auto_arm:
                self._do_arm_and_takeoff()
            self._do_lawnmower()
            self._do_land()
        except Exception as e:
            print(f"[Mission] EXCEPTION: {e}")
            self._state = "DONE"

    def _do_arm_and_takeoff(self):
        self._state = "ARMING"

        # Wait for VO to start tracking before arming
        print("[Mission] Waiting for VO TRACKING status...")
        while not self._stop_evt.is_set():
            if self.vo.status == "TRACKING":
                break
            time.sleep(0.2)

        self.cmd.set_mode("GUIDED")
        time.sleep(1.0)
        self.cmd.arm()

        self._state = "TAKEOFF"
        self.cmd.takeoff(self.cfg["takeoff_alt_m"])

        # Wait until we reach takeoff altitude (via VO y-axis ≈ altitude)
        print("[Mission] Climbing...")
        t_start = time.time()
        while not self._stop_evt.is_set():
            pos, _ = self.vo.pose()
            # In VO frame: y is DOWN, so negative y = higher altitude
            alt_est = -pos[1]
            if alt_est >= self.cfg["takeoff_alt_m"] * 0.85:
                break
            if time.time() - t_start > 15.0:
                print("[Mission] Takeoff timeout — proceeding anyway")
                break
            time.sleep(0.2)

        print(f"[Mission] At altitude. Starting lawnmower pattern.")

    def _do_lawnmower(self):
        self._state = "FLYING"
        self._current_wp_idx = 0

        while self._current_wp_idx < len(self.waypoints):
            if self._stop_evt.is_set():
                break

            wp = self.waypoints[self._current_wp_idx]
            print(f"[Mission] → WP {self._current_wp_idx}/{len(self.waypoints)-1}: "
                  f"N={wp[0]:.1f} E={wp[1]:.1f} D={wp[2]:.1f}")

            self.cmd.goto_ned(*wp)
            self._wp_start_time = time.time()

            # Navigate to waypoint
            while not self._stop_evt.is_set():
                pos, _ = self.vo.pose()

                # VO pos → approximate NED (forward=z, right=x)
                north_est = pos[2]   # z_vo = forward ≈ North
                east_est  = pos[0]   # x_vo = right  ≈ East

                # Geofence check
                if not self.fence.inside(north_est, east_est):
                    self._state = "GEOFENCE_RTL"
                    self.cmd.return_to_origin(self.cfg["geofence_return_alt_m"])
                    time.sleep(5.0)  # wait to drift back inside
                    self._state = "FLYING"
                    # re-send current waypoint
                    self.cmd.goto_ned(*wp)
                    self._wp_start_time = time.time()

                # Re-send setpoint every 500ms (ArduPilot requires periodic refresh)
                if (time.time() - self._wp_start_time) % 0.5 < 0.05:
                    self.cmd.goto_ned(*wp)

                # Arrival check
                dist = _dist2d(
                    (north_est, east_est),
                    (wp[0], wp[1])
                )
                if dist <= self.cfg["wp_accept_radius_m"]:
                    print(f"[Mission] WP {self._current_wp_idx} reached (dist={dist:.2f}m)")
                    break

                # Timeout check
                if time.time() - self._wp_start_time > self.cfg["wp_timeout_s"]:
                    print(f"[Mission] WP {self._current_wp_idx} TIMEOUT — skipping")
                    break

                time.sleep(0.1)

            self._current_wp_idx += 1

        print("[Mission] Pattern complete.")

    def _do_land(self):
        self._state = "LANDING"
        # Return to origin first, then land
        self.cmd.goto_ned(0.0, 0.0, -self.cfg["flight_alt_m"])
        time.sleep(4.0)
        self.cmd.land()
        self._state = "DONE"
        print("[Mission] DONE ✓")