# landing/geofence/geofence.py

'''
Uploads a QGroundControl inclusion polygon geofence to ArduPilot via MAVLink.
Uses the MAVLink 2 mission protocol (MISSION_COUNT / MISSION_ITEM_INT) which
is what modern ArduPilot versions require. The old fence_total_send /
fence_point_send protocol was removed in newer pymavlink builds.

ArduPilot enforces the fence natively after upload — no position polling
needed from Python.

Usage:
    from landing.geofence.geofence import upload_geofence_from_plan
    upload_geofence_from_plan(master, "/path/to/csufField.plan")
'''

import json
import time
from pymavlink import mavutil


# ArduPilot FENCE_ACTION values:
#   0 = Report only
#   1 = RTL or Land
#   2 = Always Land
FENCE_ACTION = 0    # Land / RTL on breach
FENCE_MARGIN = 1.0  # metres inside polygon before breach triggers


def _load_polygon_from_plan(plan_path: str) -> list:
    '''Parse a .plan file and return the first inclusion polygon as
    a list of (lat, lon) tuples.'''
    with open(plan_path, "r") as f:
        plan = json.load(f)

    polygons = plan.get("geoFence", {}).get("polygons", [])
    for poly in polygons:
        if poly.get("inclusion", False):
            points = poly["polygon"]
            return [(pt[0], pt[1]) for pt in points]

    raise ValueError(f"No inclusion polygon found in {plan_path}")


def _configure_fence_params(master):
    '''Set ArduPilot fence parameters before uploading the polygon.'''
    params = {
        "FENCE_ENABLE": 1,
        "FENCE_TYPE":   2,       # 2 = Polygon only
        "FENCE_ACTION": FENCE_ACTION,
        "FENCE_MARGIN": FENCE_MARGIN,
    }
    for name, value in params.items():
        master.mav.param_set_send(
            master.target_system,
            master.target_component,
            name.encode(),
            float(value),
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
        )
        msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=2)
        if msg:
            print(f"[GEOFENCE] Set {name} = {value} (confirmed: {msg.param_value})")
        else:
            print(f"[GEOFENCE] Warning: no ACK for {name}")
        time.sleep(0.1)


def _flush(master, timeout=0.3):
    '''Drain queued MAVLink messages so we read fresh responses.'''
    deadline = time.time() + timeout
    while time.time() < deadline:
        master.recv_match(blocking=False)
        time.sleep(0.01)


def upload_geofence_from_plan(master, plan_path: str):
    '''Parse the .plan file and upload the inclusion polygon to ArduPilot
    using the MAVLink 2 mission protocol (MISSION_COUNT / MISSION_ITEM_INT).

    This replaces the old fence_total_send / fence_point_send protocol
    which is not available in modern pymavlink versions.
    '''
    print(f"[GEOFENCE] Loading geofence from {plan_path}")
    polygon = _load_polygon_from_plan(plan_path)
    count   = len(polygon)
    print(f"[GEOFENCE] Polygon has {count} vertices")

    # 1. Configure fence parameters first
    _configure_fence_params(master)

    # 2. Tell ArduPilot how many fence items to expect
    #    MAV_MISSION_TYPE_FENCE = 2
    _flush(master)
    master.mav.mission_count_send(
        master.target_system,
        master.target_component,
        count,
        mavutil.mavlink.MAV_MISSION_TYPE_FENCE,
    )

    # 3. ArduPilot responds with MISSION_REQUEST_INT for each item in order.
    #    We reply with the corresponding polygon vertex.
    for i in range(count):
        req     = None
        deadline = time.time() + 5.0
        while time.time() < deadline:
            msg = master.recv_match(
                type=["MISSION_REQUEST_INT", "MISSION_REQUEST"],
                blocking=True, timeout=1.0
            )
            if msg and msg.seq == i and msg.mission_type == mavutil.mavlink.MAV_MISSION_TYPE_FENCE:
                req = msg
                break

        if req is None:
            raise RuntimeError(
                f"[GEOFENCE] Timed out waiting for MISSION_REQUEST_INT seq={i}"
            )

        lat, lon = polygon[i]
        # MAV_CMD_NAV_FENCE_POLYGON_VERTEX_INCLUSION = 5001
        # param1 = total vertex count for this polygon
        master.mav.mission_item_int_send(
            master.target_system,
            master.target_component,
            i,                                              # seq
            mavutil.mavlink.MAV_FRAME_GLOBAL,               # frame
            5001,                                           # MAV_CMD_NAV_FENCE_POLYGON_VERTEX_INCLUSION
            0, 1,                                           # current, autocontinue
            count, 0, 0, 0,                                 # param1=vertex count, rest ignored
            int(lat * 1e7),                                 # x = lat in 1e7 degrees
            int(lon * 1e7),                                 # y = lon in 1e7 degrees
            0,                                              # z = altitude (ignored for fence)
            mavutil.mavlink.MAV_MISSION_TYPE_FENCE,
        )
        print(f"[GEOFENCE] Sent vertex {i}/{count - 1}: lat={lat:.7f} lon={lon:.7f}")
        time.sleep(0.05)

    # 4. Wait for MISSION_ACK confirming upload was accepted
    ack = master.recv_match(type="MISSION_ACK", blocking=True, timeout=5.0)
    if ack and ack.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
        print("[GEOFENCE] Geofence upload accepted by ArduPilot")
    elif ack:
        raise RuntimeError(
            f"[GEOFENCE] Upload rejected — MAV_MISSION_RESULT={ack.type}"
        )
    else:
        print("[GEOFENCE] Warning: no MISSION_ACK received — fence may not have uploaded")

    print("[GEOFENCE] Geofence upload complete. ArduPilot will enforce on breach.")