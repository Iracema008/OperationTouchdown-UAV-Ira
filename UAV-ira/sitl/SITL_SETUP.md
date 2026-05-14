# SITL Setup Guide — UAV-ira

Software In The Loop (SITL) lets us test the full UAV pipeline without flying. With ArduPilot running, simulating a virtual drone, the OAK-D S2 camera provides real VO, and QGroundControl lets us watch the mission execute on a satellite map view.

---

## What you need

- Mac with Intel chip (Apple Silicon works too, but needs Rosetta for some steps)
- OAK-D S2 camera
- QGroundControl installed (download at https://qgroundcontrol.com)
- Python 3.10 or 3.11
- Activatedt venv (`rtx_venv`)

---

## Part 1 — Install dependencies

Open a terminal and run these one at a time:

```bash
brew install gcc python3 git cmake pkg-config opencv
pip3 install pymavlink MAVProxy dronekit
```

Verify cmake installed correctly:
```bash
cmake --version
# Should print: cmake version 4.x.x
```

---

## Part 2 — Clone and build ArduPilot

```bash
cd ~
git clone https://github.com/ArduPilot/ardupilot.git
cd ardupilot
git submodule update --init --recursive
```

The submodule step downloads many files. It will take a while; just wait until it returns to the prompt.

Run the ArduPilot setup script:
```bash
Tools/environment_install/install-prereqs-mac.sh -y
source ~/.zshrc
```

---

## Part 3 — Launch SITL

Replace the coordinates with an actual field location.

```bash
cd ~/ardupilot
sim_vehicle.py -v ArduCopter --console \
  --custom-location=YOUR_LAT,YOUR_LON,ALTITUDE,0 \
  --out=127.0.0.1:14550 \
  --out=127.0.0.1:14551 \
  --out=127.0.0.1:14552
```

For instance:
```bash
sim_vehicle.py -v ArduCopter --console \
  --custom-location=34.10468, -118.31910,50,0 \
  --out=127.0.0.1:14550 \
  --out=127.0.0.1:14551 \
  --out=127.0.0.1:14552
```

The last `0` is the launch heading in degrees — 0 means facing North. This is important because the  mission assumes a NORTH FACING launch.

Wait for the MAVProxy console to show:
```
STABILIZE>
```

---

## Part 4 — Configure ArduPilot Parameters

In the MAVProxy console (`STABILIZE>`) type each of these and press Enter after each one:

```
param set GPS1_TYPE 0
param set ARMING_OPTIONS 0
param set VISO_TYPE 1
param set EK3_SRC1_POSXY 6
param set EK3_SRC1_VELXY 6
param set EK3_SRC1_POSZ 1
param set EK3_SRC1_VELZ 0
param set EK3_SRC1_YAW 6
param set SIM_GPS1_ENABLE 0
```

Verify they saved correctly:
```
param show GPS1_TYPE
param show VISO_TYPE
param show EK3_SRC1_POSXY
param show SIM_GPS1_ENABLE
```

Expected output:
```
GPS1_TYPE        0.0
VISO_TYPE        1.0
EK3_SRC1_POSXY   6.0
SIM_GPS1_ENABLE  0.0
```

Save and reboot:
```
param save mav.parm
reboot
```

Wait for `STABILIZE>` to come back after reboot — about 30 seconds.

---

## Part 5 — Open a second terminal. This connects the real OAK-D S2 camera to SITL:

```bash
cd ~/path/to/UAV-ira
PYTHONPATH=. python3 sitl/sitl_vo_bridge.py
```

**Important — point the OAK-D S2 camera at a textured surface** before running this. A keyboard, printed page, or any surface with detail works. The camera needs visual features to track position (Ex: camera calibration squares).

Wait for this output:
```
[SITLBridge] Heartbeat OK — streaming at 30.0 Hz.
[SITLBridge] Vision stream thread started.
[SITLBridge] Yaw aligned. offset=+X.XX°
[SITLBridge] N=+0.00 E=+0.00 D=+0.00 yaw=+0.0° | msgs=150
```

The `Yaw aligned` line is the key — it means ArduPilot has accepted the VO feed.

**Do not move on until you see `Yaw aligned`.**

---

## Part 6 — Connect QGroundControl

Open QGroundControl. It should connect automatically on UDP port 14550.

You should see:
- The virtual drone on the map at your field coordinates
- No red warning icons
- Status showing **Ready to Fly**

If QGC still shows `PreArm: VisOdom: not healthy` it means the bridge isn't streaming yet — go back to Part 5 and make sure `Yaw aligned` appeared.

---

## Part 7 — Arm and Run The Mission

**In Terminal 1 (MAVProxy):**
```
mode guided
arm throttle
takeoff 2
```

Watch QGC — the virtual drone lifts off to 2 meters.

**In Terminal 3 (new terminal):**
```bash
cd ~/path/to/UAV-ira
PYTHONPATH=. python3 - <<'EOF'
import sys, os, time, math
sys.path.insert(0, os.getcwd())
from pymavlink import mavutil

master = mavutil.mavlink_connection('udpout:127.0.0.1:14552')
master.wait_heartbeat()
print('Connected to SITL')

def generate_waypoints():
    wps = []
    alt = -2.0
    right = 0.0
    direction = 1
    while right <= 8.0 + 1e-6:
        n_near = 0.0 if direction == 1 else 8.0
        n_far  = 8.0 if direction == 1 else 0.0
        wps.append((n_near, right, alt))
        wps.append((n_far,  right, alt))
        right += 1.0
        direction *= -1
    return wps

def goto_ned(master, north, east, down):
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
    master.mav.set_position_target_local_ned_send(
        0, master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask, north, east, down,
        0, 0, 0, 0, 0, 0, 0, 0
    )

waypoints = generate_waypoints()
print(f'Running lawnmower — {len(waypoints)} waypoints')

for i, wp in enumerate(waypoints):
    print(f'WP {i:02d}/{len(waypoints)-1}: N={wp[0]:.1f} E={wp[1]:.1f} D={wp[2]:.1f}')
    t_start = time.time()
    while time.time() - t_start < 8.0:
        goto_ned(master, *wp)
        msg = master.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=1)
        if msg:
            dist = math.sqrt((msg.x - wp[0])**2 + (msg.y - wp[1])**2)
            print(f'  pos: N={msg.x:+.2f} E={msg.y:+.2f} dist={dist:.2f}m', end='\r')
            if dist < 0.5:
                print(f'\n  Arrived at WP {i}')
                break
        time.sleep(0.2)

print('\nMission complete!')
EOF
```

Watch the QGC map — the virtual drone traces the lawnmower pattern over your field.

---

## Terminal layout summary

| Terminal | What runs | Notes |
|---|---|---|
| 1 | `sim_vehicle.py` — SITL + MAVProxy | Never close this (or else) |
| 2 | `sitl_vo_bridge.py` — VO + ArUco | Keep OAK-D S2 pointed at a textured surface |
| 3 | Mission script | Can now restart freely |
| QGC | QGroundControl | Opens automatically |

---

## Troubleshooting

**`PreArm: VisOdom: not healthy`**
The bridge isn't running or VO hasn't reached TRACKING status. Make sure the OAK-D is pointed at a textured surface and `Yaw aligned` has appeared in Terminal 2.

**`Unable to find parameter 'GPS_TYPE'`**
You're on ArduPilot v4.8+. Use `GPS1_TYPE` instead.

**`X_LINK_INSUFFICIENT_PERMISSIONS`**
The OAK-D is locked from a previous crashed session. Unplug, wait 5 seconds, replug.

**`X_LINK_ERROR` / USB disconnect**
Cable quality issue. Use a short thick USB 3.0 cable and try the other Thunderbolt port on your Mac.

**Bridge connects but `Yaw aligned` never appears**
VO is stuck in WARMUP. Point the camera at a surface with more texture and depth variation — a keyboard works better than a flat printed page. Hold the camera 40-50cm above the surface.

**`inliers=0` in the camera window**
Stereo depth isn't working for the current surface. Move the camera closer (minimum 20cm, optimal 40-80cm) or point at a scene with objects at different depths.

**Mission script hangs on `wait_heartbeat`**
Port 14552 isn't open. Make sure you started SITL with all three `--out` flags as shown in Part 3.

---

## ArUco marker detection

While the bridge is running, the system automatically detects ArUco markers (DICT_6X6_250) from the OAK-D S2 RGB camera.

**Valid marker IDs** are set in `sitl/sitl_vo_bridge.py`:
```python
TARGET_IDS = [3, 7]
```

Edit this list based on which marker IDs is the target you've placed on your field.

When a valid marker is detected:
- Terminal 2 prints `[ArUco] Correct marker FOUND: [3]`
- A grayscale image is saved to `saved-aruco/pack_image.png`
- Invalid IDs are detected and drawn on screen but nothing is saved

To test detection, print an ArUco marker from https://chev.me/arucogen/ using Dictionary `6x6` and hold it in front of the camera while the bridge is running.
