### UAV System:

1. Drone takes off and scans the field using its OAK camera
2. Vision system detects the target ArUco marker and records its location
3. Drone hovers and waits for the UGV to signal it's ready
4. Drone locks onto the april tag on top of the UGV and lands on it

## File Structure:

```
UAV-ira/
    main.py
      - core/         # Shared state, config, and threading utils
      - vision/       # ArUco marker detection
      - landing/      # April tag detection and Pixhawk flight commands
      - slam/         # Visual odometry, tracks where the drone is
      - comms/        # UGV communication
      - tests/        # Hardware and flight test scripts
```


## Shared State:

All packages communicate through a single `UAVState` object in `core/state.py`. No package will talk directly to another, they only read and write the shared state, and use events to signal transitions (for instance, marker found, call for UGV).


## How to Run:

```bash
# Full run
python3 main.py --mode scan

# Test landing only
python3 main.py --mode land
```

## Hardware List:
    - DepthAI OAK camera (vision, slam )
    - Pixhawk (MAVLink) (landing )
    - Raspberry Pi 5 UART (comms )
