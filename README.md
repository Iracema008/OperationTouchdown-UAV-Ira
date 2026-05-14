# UAV System Overview

1. Drone takes off and scans the field using its OAK camera
2. Vision system detects the target ArUco marker and records its location
3. Drone executes a search pattern (lawnmower)
4. Once detected, the drone transitions to landing mode
5. Drone detects the AprilTag on the UGV
6. Drone aligns and lands on the UGV

## File Structure

```
UAV-ira/
    main.py
    core/         # Shared state, config, logging
    vision/       # ArUco detection + segmentation
    mision/      # AprilTag landing + Pixhawk control
    vio_slam/     # VO + loop closure
    sitl/         # Flight simmulation w/ArduPilot

    tests/        # Hardware testing scripts (WIP)
    comms/        # UGV communication (WIP)

README.md
requirements.txt
.gitignore
```


## Shared State

All modules communicate through a single `UAVState` object in `core/state.py`.
No module directly calls another. Threads read/write shared state and use events for transitions (for instance, marker detected --> landing).


## Requirements

### Hardware

* DepthAI OAK-D S2 camera
* Pixhawk 6X (ArduPilot via QGroundControl)
* Raspberry Pi 5 or onboard computer

### Software

* Python 3.10+

## Setup

### 1. Clone repository

```bash
git clone https://github.com/CSUF-RAYTHEON/OperationTouchdown.git
cd OperationTouchdown
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



## Configuration

Configuration is defined in:

```
core/config.py
```

Key defaults:

* Pixhawk connection: `/dev/serial0`
* Baud rate: `57600`
* Hover altitude: `5.0 m`
* ArUco target ID: '3, 7`

Override Pixhawk port if needed:

```bash
export PIXHAWK_PORT=/dev/ttyUSB0
```

---

## How to Run

Navigate to the main module:

```bash
cd UAV-ira
```

Run full mission:

```bash
python3 main.py --mode scan
```

Run landing only:

```bash
python3 main.py --mode land
```

## How to Run Simulator
* Go into sitl folder for an in-depth explanation (SITL_SETUP.md)

## Runtime Behavior

* System arms and takes off
* SLAM thread estimates position
* Vision detects ArUco marker
* Lawnmower search runs until detection
* Landing thread uses AprilTag for alignment and descent


## Notes

* Uses DepthAI pipeline for stereo + inertial measurement unit (IMU)
* MAVLink used for Pixhawk control
* Multi-threaded with shared state coordination
* Alan said something about concurrency but i forgor


## Troubleshooting
**DepthAI not detected**

```bash
python3 -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"
```
**Luxonis Camera Troubleshooting**
If you run into this error:
"[warning] Insufficient permissions to communicate with X_LINK_UNBOOTED device with name "3.1". Make sure udev rules are set...” 

1.
   `echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules`
3.
   `sudo udevadm control --reload-rules && sudo udevadm trigger`


**Pixhawk not connecting**

* Check port (`/dev/ttyAMA0` or `/dev/ttyUSB0`)
* Ensure baud rate is 57600

**Permission issues**

```bash
sudo usermod -aG dialout $USER
```

(then restart)

