# landing/geofence/geofence.py

"""
Uploads a QGroundControl inclusion polygon geofence to ArduPilot via MAVLink.
ArduPilot enforces the fence natively — no position polling needed on our side.

Breach action is configured to RTL (Return to Land), which for a GPS-denied
indoor system effectively means the FC triggers a land-in-place since there's
no home GPS lock. You can change FENCE_ACTION to 1 (Land) if preferred.

Usage:
    from landing.geofence.geofence import upload_geofence_from_plan
    upload_geofence_from_plan(master, "/path/to/30yard.plan")
"""

import json
import time
from pymavlink import mavutil


# ArduPilot FENCE_ACTION values:
#   0 = Report only
#   1 = RTL or Land
#   2 = Always Land
#   3 = SmartRTL or Land
FENCE_ACTION = 1   # Land / RTL on breach
FENCE_MARGIN = 1.0 # meters inside the polygon to trigger breach (buffer)


def _load_polygon_from_plan(plan_path: str) -> list:
    """
    Parse a .plan file and return the first inclusion polygon as
    a list of (lat, lon) tuples.
    """
    with open(plan_path, "r") as f:
        plan = json.load(f)

    polygons = plan.get("geoFence", {}).get("polygons", [])
    for poly in polygons:
        if poly.get("inclusion", False):
            points = poly["polygon"]
            return [(pt[0], pt[1]) for pt in points]

    raise ValueError(f"No inclusion polygon found in {plan_path}")


def _configure_fence_params(master):
    """
    Set ArduPilot fence parameters before uploading the polygon.
    """
    params = {
        "FENCE_ENABLE": 1,
        "FENCE_TYPE":   2,      # 2 = Polygon only
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
        # Wait for ACK
        msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=2)
        if msg:
            print(f"[GEOFENCE] Set {name} = {value} (confirmed: {msg.param_value})")
        else:
            print(f"[GEOFENCE] Warning: no ACK for {name}")
        time.sleep(0.1)


def upload_geofence_from_plan(master, plan_path: str):
    """
    Parse the .plan file and upload the inclusion polygon to ArduPilot.
    Call this once after arming params are set, before takeoff.
    """
    print(f"[GEOFENCE] Loading geofence from {plan_path}")
    polygon = _load_polygon_from_plan(plan_path)
    count   = len(polygon)
    print(f"[GEOFENCE] Polygon has {count} vertices")

    # 1. Configure fence parameters
    _configure_fence_params(master)

    # 2. Send total fence point count (ArduPilot needs this first)
    master.mav.fence_total_send(count + 1)  # +1 for the mandatory closing point
    time.sleep(0.2)

    # 3. Upload each vertex
    for i, (lat, lon) in enumerate(polygon):
        master.mav.fence_point_send(
            master.target_system,
            master.target_component,
            i,          # point index
            count + 1,  # total count including closing point
            lat,
            lon,
        )
        print(f"[GEOFENCE] Sent vertex {i}: lat={lat:.7f} lon={lon:.7f}")
        time.sleep(0.15)  # small gap so FC doesn't drop packets

    # 4. Send closing point (must duplicate vertex 0 to close the polygon)
    close_lat, close_lon = polygon[0]
    master.mav.fence_point_send(
        master.target_system,
        master.target_component,
        count,          # last index
        count + 1,
        close_lat,
        close_lon,
    )
    print(f"[GEOFENCE] Sent closing point (duplicate of vertex 0)")

    # 5. Verify upload by reading back point 0
    master.mav.fence_fetch_point_send(
        master.target_system,
        master.target_component,
        0,
    )
    verify = master.recv_match(type="FENCE_POINT", blocking=True, timeout=2)
    if verify:
        print(f"[GEOFENCE] Verified vertex 0: lat={verify.lat:.7f} lon={verify.lng:.7f}")
    else:
        print("[GEOFENCE] Warning: could not verify upload — check FC connection")

    print("[GEOFENCE] Geofence upload complete. ArduPilot will enforce on breach.")