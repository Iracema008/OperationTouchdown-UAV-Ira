# UAV System Overview

1. Drone takes off and scans the field using its OAK-D S2 camera
2. VIO system estimates position in GPS-denied environment using optical flow
3. SLAM corrects drift via loop closure
4. Vision detects target ArUco marker and records its NED location
5. Drone executes a search pattern (lawnmower grid or simulated annealing)
6. Once detected, drone transitions to landing mode
7. Drone detects AprilTag on top of moving UGV
8. Drone aligns, tracks, and lands on the moving UGV

---

## Setup

### 1. Clone repository

```bash
git clone https://github.com/CSUF-RAYTHEON/OperationTouchdown.git
cd OperationTouchdown/UAV-ira
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set PYTHONPATH

Add to `~/.bashrc` on the Pi so it persists across terminals:

```bash
export PYTHONPATH=/home/pi/UAV-ira
```

---

## How to Run

### Real flight

```bash
cd UAV-ira
python main.py --mode scan --planner grid     # boustrophedon lawnmower
python main.py --mode scan --planner sa       # simulated annealing lawnmower
python main.py --mode scan --planner c1       # challenge 1
python main.py --mode scan --planner c2       # challenge 2
```

### SITL (Mac, no drone)

See `SITL_SETUP.md` for full setup. Quick start once SITL is running:

```bash
export PYTHONPATH=/Users/sema/Documents/OperationTouchdown/UAV-ira
python sitl/run_sitl.py --planner grid
python sitl/run_sitl.py --planner sa
```

Press Enter at any point during SITL to simulate `marker_confirmed` and test the landing sequence.

---

## Configuration

All configuration is in `core/config.py`. Key defaults:

| Setting | Default | Notes |
|---|---|---|
| Pixhawk port | `/dev/serial0` | Change to `udpin:0.0.0.0:14550` for SITL |
| Baud rate | `57600` | All UARTs |
| Hover altitude | `3.0m` | Search altitude during lawnmower |
| ArUco target IDs | `[3, 7]` | Edit for your marker IDs |
| Field size | `8x8m` | Set in each mission file |
| Column spacing | `2.8m` | Derived from OAK-D FOV at 3m altitude |
| UGV serial port | `/dev/ttyUSB0` | LoRa dongle on Pi |

Override Pixhawk port at runtime if needed:

```bash
export PIXHAWK_PORT=/dev/ttyUSB0
```

---

## Mission Planners

**grid** — plain boustrophedon column search. Visits waypoints in fixed left-to-right alternating up-down order. Simple and predictable.

**sa** — simulated annealing optimized search. Runs once before arming and reorders the same grid waypoints into the shortest total path. On an 8x8m field SA typically saves 10-25% path distance over grid. Full optimized path is printed to logs before takeoff.

**c1** — Challenge 1. Takes off from stationary UGV, flies forward until AprilTag on UGV is detected, lands on moving UGV, disarms, waits 30 seconds.

**c2** — Challenge 2. Takes off, runs grid search to find ArUco marker, sends NED coordinates to UGV over LoRa, returns to start, detects AprilTag on moving UGV, lands, waits 10 seconds.

---

## File Structure

```
UAV-ira/
    main.py                         Entry point for all real flight missions
    core/
        config.py                   Camera, Pixhawk, UART, ArUco, field config
        state.py                    Shared memory creation and UAV state accessor
        log.py                      Logging setup
    controls/
        connect.py                  UART0/2/3 MAVLink connection helpers
        attitude.py                 Request and read attitude messages
        nedlocalposition.py         Request and read LOCAL_POSITION_NED messages
        busywait.py                 Precise short-duration delay
        ugv_comms.py                Send goto / stop / drive commands to UGV over LoRa
        lora_sender.py              Manual LoRa send tool (Mac, for testing)
    mission/
        mission_grid.py             Boustrophedon lawnmower search + ArUco + landing
        mission_sa.py               Simulated annealing search + ArUco + landing
        challenge_one.py            Challenge 1 — launch, fly forward, land on moving UGV
        challenge_two.py            Challenge 2 — search, find marker, signal UGV, land on UGV
    path_planning/
        lawnmower.py                Grid waypoint generation
        simulated_annealing.py      SA path optimization + pre-flight reorder
    landing/
        stationary_landing.py       AprilTag detection and landing pipeline (standalone)
        pixhawk_controller/
            stationary_landing_controller.py    MAVLink arm/fly/land/velocity controller
    vio_slam/
        broadcaster.py              Sole OAK-D owner — writes frames and attitude to shared memory
        vio.py                      Optical flow + PnP position estimation, sends to Pixhawk
        slam.py                     ORB loop closure drift correction
        viewer.py                   Debug viewer for shared memory frames
    vision/
        detectors/
            detector_manager.py     Selects and returns the configured detector
            detector.py             Base detector class
            available_detectors.py  Detector registry
            opencv_helpers.py       CV2 ArUco utilities
            april_tag_detector.py   AprilTag 36h11 pose estimation
        video/
            camera_coordinate_transformer.py    Pixel to NED metre conversion
        segmentation/
            uav_segmenter.py        Field segmentation (WIP)
    telemetry/
        telemetry_logger.py         Logs flight data to SQLite
        health_monitor.py           Process health monitoring
        post_flight_export.py       Export flight logs to CSV
    sitl/
        run_sitl.py                 SITL test runner — see SITL_SETUP.md
        deprecated/                 Old bridge files, kept for reference
README.md
SITL_SETUP.md
requirements.txt
.gitignore
```

---

## Process Architecture

Five processes run in parallel during a real flight. Each owns exactly one hardware resource.

| Process | File | UART | Purpose |
|---|---|---|---|
| broadcaster | vio_slam/broadcaster.py | UART2 | Sole OAK-D owner. Writes frames + attitude to shared memory |
| vio | vio_slam/vio.py | UART3 | Reads frames, computes NED position, sends to Pixhawk |
| slam | vio_slam/slam.py | — | Loop closure drift correction |
| mission | mission/*.py | UART0 | ArUco/AprilTag detection, path planning, landing |
| telemetry | telemetry/telemetry_logger.py | — | Logs flight data to SQLite |

SITL runs only the mission process. Broadcaster, VIO, and SLAM are skipped since there is no Pixhawk or OAK-D on your Mac.

---

## Shared State

All processes communicate through shared memory blocks managed in `core/state.py`. No process calls another directly. Processes read and write named shared memory blocks and use multiprocessing Events for phase transitions.

```
uav_vio      [north, east, down, yaw, timestamp]   VIO writes, mission reads
uav_aruco    [x, y, z, id, timestamp]              Mission writes on confirmation
oak_rgb      [H x W x 3 uint8]                     Broadcaster writes, mission reads
oak_gray     [H x W uint8]                         Broadcaster writes, VIO reads
oak_depth    [H x W uint16]                        Broadcaster writes, VIO reads
oak_calib    [3x3 float64]                         Broadcaster writes once at startup
attitude     [roll, pitch, yaw]                    Broadcaster writes, VIO reads
position     [north, east, down]                   VIO writes, SLAM reads
```

---

## Hardware Requirements

- OAK-D S2 camera (DepthAI v3)
- Pixhawk flight controller (ArduCopter firmware)
- Raspberry Pi 4 (4GB or 8GB)
- SB Components 915MHz USB LoRa dongle (for UGV communication)
- Long USB-C cable (for bench testing with camera bolted to drone)

---

## UART Wiring

Three separate UART connections to the Pixhawk are required. Each process owns exactly one port.

| Pi UART | Device | Process | Purpose |
|---|---|---|---|
| UART2 | /dev/ttyAMA2 | broadcaster | Reads attitude from Pixhawk |
| UART3 | /dev/ttyAMA3 | vio | Sends vision_position_estimate to Pixhawk |
| UART0 | /dev/serial0 | mission | Sends arm / goto / land commands |

Enable UART2 and UART3 on the Pi by adding to `/boot/config.txt`:

```
dtoverlay=uart2
dtoverlay=uart3
```

Then reboot the Pi.

---

## Camera Intrinsics

OAK-D S2 intrinsics at 640x400 resolution (calibrated on hardware):

```
fx = 576.18    fy = 575.93
cx = 311.67    cy = 214.92
```

Set in `core/config.py`. Run this on the Pi to get exact values for your specific camera:

```bash
python3 -c "
import depthai as dai
d = dai.Device()
K = d.readCalibration().getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 400)
print(f'fx={K[0][0]:.2f} fy={K[1][1]:.2f} cx={K[0][2]:.2f} cy={K[1][2]:.2f}')
"
```

---

## ArUco Markers

Target marker IDs are set in `core/config.py` under `ArucoConfig.target_marker_id`. Default: `[3, 7]`.

The system requires 3 consecutive detections to confirm a marker. Detection runs every frame at 30Hz in the mission process — no DepthAI pipeline opened in mission, reads RGB from shared memory written by broadcaster.

Print markers from https://chev.me/arucogen using Dictionary `6x6_250`. Physical marker size: 20cm (`marker_size_m = 0.2` in config).

---

## AprilTag Landing

Target tag ID 67, family `tag36h11`, size 20cm. Landing logic in `landing/stationary_landing_controller.py`:

- PID velocity control to align above tag
- Gain scheduling based on altitude (aggressive far, gentle close)
- `smart_touchdown` detects physical contact via vertical speed flatlining
- Disarms immediately on touchdown

Precision landing is enabled in challenge missions. In grid and SA missions it is commented out pending lawnmower validation — uncomment `_run_apriltag_landing()` in the land phase when ready.

---

## UGV Communication

Commands are sent over a 915MHz LoRa serial link to the UGV ROS2 bridge node.

| Function | Sends | Effect |
|---|---|---|
| `send_goto(north, east, cfg)` | `{"x": 2.5, "y": 1.0}` | UGV navigates to NED coordinates |
| `send_drive_c1(cfg)` | `STRAIGHT` | UGV drives straight for C1 |
| `send_stop(cfg)` | `STOP` | UGV stops immediately |

UGV must face north at the start of each run. This aligns the UGV body frame with the UAV NED frame so coordinates are correct without rotation.

---

## Troubleshooting

**DepthAI not detected**

```bash
python3 -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"
```

**Insufficient permissions for OAK-D**

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

**`X_LINK_DEVICE_NOT_FOUND`**
OAK-D locked from a previous crashed session. Unplug, wait 5 seconds, replug.

**Pixhawk not connecting**
- Check port is correct (`/dev/ttyAMA0`, `/dev/ttyAMA2`, `/dev/ttyAMA3`)
- Verify baud rate is 57600
- Only one process per UART — check nothing else has the port open

**`No module named 'core'`**
PYTHONPATH not set. Run:
```bash
export PYTHONPATH=/home/pi/UAV-ira
```

**Permission denied on serial port**

```bash
sudo usermod -aG dialout $USER
```

Then log out and back in.

**LoRa dongle not found**

```bash
ls /dev/ttyUSB*
```

Update `cfg.comms.serial_port` in `core/config.py` if it landed on a different port.

**VIO status shows `LOW_TRACK` or `DEPTH_FILTER`**
Not enough visual features. Point camera at a textured surface. Minimum depth 8cm, maximum 18m.

**`PreArm: VisOdom: not healthy` (SITL only)**
Set `VISO_TYPE 0` in MAVProxy console — see `SITL_SETUP.md`.

---

## Notes

- Uses DepthAI v3 pipeline for stereo + IMU
- MAVLink used for all Pixhawk control
- Multiprocessing with shared memory — one process per hardware resource
- Alan said something about concurrency but i forgor