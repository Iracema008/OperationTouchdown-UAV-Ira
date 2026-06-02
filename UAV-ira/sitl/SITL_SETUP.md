# SITL Setup Guide — UAV-ira

Software In The Loop (SITL) lets you test the full mission pipeline without flying. ArduPilot simulates a virtual drone, and the mission process arms, sweeps, and lands against real MAVLink — no camera or Pixhawk required.

---

## What you need

- Mac (Intel or Apple Silicon)
- Python 3.11
- Activated virtual environment (`rtx_venv`)
- ArduPilot + MAVProxy installed
- QGroundControl (optional, for watching the mission on a map)

---

## Part 1 — Install ArduPilot

```bash
cd ~
git clone https://github.com/ArduPilot/ardupilot.git
cd ardupilot
git submodule update --init --recursive
Tools/environment_install/install-prereqs-mac.sh -y
source ~/.zshrc
```

The submodule step downloads many files and will take a while. Just wait until it returns to the prompt.

Install MAVProxy:

```bash
pip install MAVProxy
```

---

## Part 2 — Set PYTHONPATH

Every terminal you open for SITL testing needs this set:

```bash
export PYTHONPATH=/Users/sema/Documents/OperationTouchdown/UAV-ira
```

Add it to `~/.zshrc` to make it permanent:

```bash
echo 'export PYTHONPATH=/Users/sema/Documents/OperationTouchdown/UAV-ira' >> ~/.zshrc
source ~/.zshrc
```

---

## Part 3 — Launch SITL

Replace the coordinates with your actual field location. The final `0` is the launch heading in degrees — 0 means facing North. The mission assumes a north-facing launch.

```bash
cd ~/ardupilot
sim_vehicle.py -v ArduCopter --console \
  --custom-location=34.10468,-118.31910,50,0
```

Wait for the MAVProxy console to show:

```
STABILIZE>
```

---

## Part 4 — Configure ArduPilot parameters

In the MAVProxy console type each of these and press Enter after each one:

```
param set VISO_TYPE 0
param set EK3_SRC1_POSXY 3
param set EK3_SRC1_VELXY 3
param set EK3_SRC1_POSZ 1
param set ARMING_CHECK 0
```

What these do:

- `VISO_TYPE 0` — disables the visual odometry requirement. VIO only runs on the Pi, not your Mac.
- `EK3_SRC1_POSXY 3` — tells the EKF to use simulated GPS for horizontal position instead of VIO.
- `EK3_SRC1_VELXY 3` — same for velocity.
- `EK3_SRC1_POSZ 1` — use barometer for altitude.
- `ARMING_CHECK 0` — disables pre-arm checks so the drone arms without sensors that aren't present on your Mac.

Save the parameters so you don't have to retype them next session:

```
param save sitl_params.parm
```

Load them next time with:

```
param load sitl_params.parm
```

Wait until the console shows `pre-arm good` before running the mission script.

---

## Part 5 — Run the mission

Open a second terminal and activate your environment:

```bash
cd /Users/sema/Documents/OperationTouchdown/UAV-ira
source rtx_venv/bin/activate
```

Run the grid lawnmower:

```bash
python sitl/run_sitl.py --planner grid
```

Run the simulated annealing lawnmower:

```bash
python sitl/run_sitl.py --planner sa
```

The mission process arms the drone, takes off to 3m, and begins the sweep. Position is read directly from SITL's `LOCAL_POSITION_NED` messages — no VIO or camera required.

Press Enter at any point to simulate `marker_confirmed` and test the approach and landing sequence.

---

## What you will see

Grid planner startup:

```
[GRID] === PLANNED PATH ===
[GRID]   WP  1/20 → N=0.0 E=0.0
[GRID]   WP  2/20 → N=2.0 E=0.0
...
[GRID] Sweep starting — 20 waypoints, dwell=3.0s each
```

SA planner startup (runs optimisation before arming):

```
[SA] Running pre-flight path optimisation...
[SA] 20 waypoints — 38.4m → 31.2m over 85 steps
[SA] === OPTIMISED PATH ===
[SA]   WP  1/20 → N=0.0 E=0.0
[SA]   WP  2/20 → N=8.0 E=5.6
...
```

On each waypoint departure:

```
[GRID] Departing WP 3/20 | from N=0.02 E=0.00 | to N=4.0 E=0.0 | dist to next=4.00m
```

Waypoints will all time out and advance on a fixed dwell timer — this is expected. SITL has no physics simulation of the drone actually moving, so position stays near the origin. The dwell timer advances waypoints regardless. This is correct for testing path logic and MAVLink commands.

---

## Terminal layout

| Terminal | What runs |
|---|---|
| 1 | `sim_vehicle.py` — SITL + MAVProxy. Never close this during a test. |
| 2 | `run_sitl.py` — mission script. Can be restarted freely. |

QGroundControl can be open at the same time and will connect automatically on UDP port 14550.

---

## Troubleshooting

**`PreArm: VisOdom: not healthy`**
Run the parameter set commands from Part 4. This means ArduPilot is waiting for VIO input that only exists on the Pi.

**`PreArm: AHRS: waiting for home`**
The simulated GPS hasn't locked yet. Wait 15-20 seconds after `pre-arm good` appears before running the mission script.

**`Address already in use` on port 14550**
A previous run left the port open. Find and kill it:
```bash
lsof -i udp:14550
kill <PID>
```

**`No module named 'core'`**
PYTHONPATH is not set. Run:
```bash
export PYTHONPATH=/Users/sema/Documents/OperationTouchdown/UAV-ira
```

**`'PixhawkConfig' object has no attribute 'connection_string'`**
Your local `core/config.py` uses a different field name. Check what the connection string field is called:
```bash
grep -n "connection\|serial\|port" core/config.py
```
Update `mission_grid.py` and `mission_sa.py` to match.

**`Waiting for heartbeat` hangs forever**
The mission is trying to connect on the wrong port. Make sure `connection_string` in `core/config.py` is set to:
```
udpin:0.0.0.0:14550
```

**`Failed to set ARMING_CHECK`**
Non-critical. The drone still arms. Verify with `param show ARMING_CHECK` if needed.

**Waypoints all timing out**
Expected. See note above about SITL having no physics simulation.