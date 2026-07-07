# UAV Preflight Checklist — OperationTouchdown

**Aircraft:** UAV
**Date:** ______________  **Location:** ______________
**Pilot in Command:** ______________  **Safety Observer:** ______________

> I recommend completing each section in order. Do not proceed to the next section until all items
> in the current section are marked PASS. Any FAIL requires resolution before
> continuing — do not arm on an unresolved FAIL. READ THE KILL SWITCH COMMANDS.
> I listed only the most critical joystick positions for the TX16S remotes. 

---

## Section 1 — Software Configuration Verification

Complete before leaving for the field, or immediately on arrival before hardware setup.

| # | Check | Command | Expected Result | Pass/Fail |
|---|-------|---------|------------------|-----------|
| 1.1 | Pull latest code | `cd /home/uav/Desktop/OperationTouchdown && git pull origin UAV-ira` | Up to date, no conflicts | ☐ |
| 1.2 | Hardware mode (not SITL) | `grep "connection_string" UAV-ira/core/config.py` | `/dev/serial0`, **not** `udp` | ☐ |
| 1.3 | ArUco target IDs | `grep "target_marker_id" UAV-ira/core/config.py` | Matches competition-assigned IDs | ☐ |
| 1.4 | Field dimensions | `grep "north_max\|east_max" UAV-ira/mission/mission_grid.py` | Matches measured field (~9.2) | ☐ |

**Section 1 Result:** ☐ GO ☐ NO-GO

---

## Section 2 — Hardware Inspection (Power Off)

| # | Check | Pass/Fail |
|---|-------|-----------|
| 2.1 | Propeller guards mounted and tight on all four arms | ☐ |
| 2.2 | All propellers firmly seated, spin freely by hand | ☐ |
| 2.3 | Frame and arms free of visible cracks or damage | ☐ |
| 2.4 | Battery fully charged and securely seated | ☐ |
| 2.5 | TELEM cable: Pixhawk → Raspberry Pi connected | ☐ |
| 2.6 | OAK-D USB-C → Raspberry Pi connected | ☐ |
| 2.7 | LoRa dongle seated in Raspberry Pi USB port | ☐ |
| 2.8 | OAK-D mounted firmly, oriented downward/forward | ☐ |
| 2.9 | No loose wires within propeller strike radius | ☐ |
| 3.0 | Check the power distribution board didn't disconnect from the battery | ☐ |

**Section 2 Result:** ☐ GO ☐ NO-GO

---

## Section 3 — Power-On Systems Verification

Power on Pixhawk and Raspberry Pi. Run each check individually; do not proceed on any FAIL.

**3.1 — UART Port Connectivity**
```bash
python3 -c "
from pymavlink import mavutil
for port in ['/dev/serial0', '/dev/ttyAMA2', '/dev/ttyAMA3']:
    try:
        m = mavutil.mavlink_connection(port, baud=57600)
        m.wait_heartbeat(timeout=5)
        print(f'{port} OK')
        m.close()
    except Exception as e:
        print(f'{port} FAILED: {e}')
"
```
Expected: `OK` on all three ports. ☐ Pass ☐ Fail

**3.2 — OAK-D Camera Detection**
```bash
python3 -c "import depthai as dai; devs = dai.Device.getAllAvailableDevices(); print(f'OAK-D devices: {len(devs)}'); print('OK' if devs else 'NOT FOUND')"
```
Expected: Device count ≥ 1, `OK`. ☐ Pass ☐ Fail

**3.3 — Position Stream (LOCAL_POSITION_NED)**
```bash
python3 -c "
from pymavlink import mavutil
m = mavutil.mavlink_connection('/dev/serial0', baud=57600)
m.wait_heartbeat()
msg = m.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=5)
print(f'Position OK: N={msg.x:.2f} E={msg.y:.2f} D={msg.z:.2f}' if msg else 'No position — check SERIAL params')
"
```
Expected: Valid N/E/D values returned. ☐ Pass ☐ Fail

**3.4 — LoRa Telemetry Dongle**
```bash
ls /dev/ttyUSB*
```
Expected: `/dev/ttyUSB0` present. ☐ Pass ☐ Fail

**Section 3 Result:** ☐ GO ☐ NO-GO

---

## Section 4 — Ground Control Station Verification (QGroundControl)

Connect Pixhawk to ground control station via USB.

| # | Check | Pass/Fail |
|---|-------|-----------|
| 4.1 | EKF status indicator: green | ☐ |
| 4.2 | No active red pre-arm warnings | ☐ |
| 4.3 | Geofence polygon uploaded and visible on map | ☐ |
| 4.4 | `SERIAL1_PROTOCOL` = 2 | ☐ |
| 4.5 | `SERIAL2_PROTOCOL` = 2 | ☐ |
| 4.6 | `SERIAL3_PROTOCOL` = 2 | ☐ |
| 4.7 | `GPS1_TYPE` = 0 (GPS disabled) | ☐ |
| 4.8 | `ARMING_CHECK` set per mission requirements | ☐ |
| 4.9 | Kill switch channel configured and function-tested | ☐ |
| 4.10 | Battery voltage reads correctly in GCS | ☐ |

**Section 4 Result:** ☐ GO ☐ NO-GO

---

## Section 5 — Flight Readiness Test (Low Hover)

Conduct before any full mission run.

1. Position aircraft at field southwest corner, nose oriented north.
2. Execute: `python main.py --mode scan --planner grid`
3. Verify arm sequence, takeoff, and stable hover at 3 m.
4. Actuate kill switch — confirm immediate, safe landing response.
5. Verify `flight_logs/flight_*.csv` was created and contains valid position rows.

| # | Check | Pass/Fail |
|---|-------|-----------|
| 5.1 | Arms and takes off normally | ☐ |
| 5.2 | Hovers stably at 3 m | ☐ |
| 5.3 | Kill switch produces safe landing | ☐ |
| 5.4 | Telemetry CSV generated with valid data | ☐ |

**Section 5 Result:** ☐ GO — cleared for full mission run ☐ NO-GO

---

## Section 6 — Transmitter & Kill Switch Configuration

**Transmitter:** RadioMaster TX16S

| # | Check | Pass/Fail |
|---|-------|-----------|
| 6.1 | Transmitter powered on and bound to receiver | ☐ |
| 6.2 | Left joystick controls throttle / yaw | ☐ |
| 6.3 | Right joystick controls pitch / roll | ☐ |
| 6.4 | Kill switch mapped to channel: ______ | ☐ |
| 6.5 | Kill switch function verified in QGC Radio Setup (channel moves as expected) | ☐ |
| 6.6 | Kill switch is within immediate reach of Safety Observer at all times | ☐ |
| 6.7 | Transmitter battery level sufficient for full flight duration | ☐ |

**Section 6 Result:** ☐ GO ☐ NO-GO

### Arming & Kill Switch Sequence (SB / SF)

The TX16S sends two independent switch positions to the flight controller — **SB** and
**SF** — each mapped to a separate function. Both switches must be verified in the
correct position *before* the propellers are commanded to spin, and returned to their
safe positions in the correct order after landing.

| Step | Switch | Position | What it does |
|------|--------|----------|---------------|
| Pre-arm | **SB** | Full up | Signals the flight controller that arming is permitted. This is a software-level arm/disarm gate — it does not cut power directly. |
| Pre-arm | **SF** | Full up | Enables motor output at the ESC level. This is the hard kill line — when down, no signal reaches the motors regardless of any other command. |
| At spin-up | — | — | The instant propellers begin spinning, move **SF down**. This keeps the hard motor-output line armed only for the moment of transition, minimizing the window where a stray command could spin the motors unexpectedly. |
| During flight | **SB** | Monitor | If an emergency landing is needed, move **SB fully down**. This immediately commands the flight controller to disarm and initiate a controlled landing/motor stop. |
| After landing | **SF** | Return to full up | This is the final, hard kill of the propellers — it cuts motor output at the ESC level independent of flight controller state. Always confirm SF is back up before approaching the aircraft or before the next flight. |

**Rule of thumb:** SB is the *software arm/disarm* — use it to trigger a safe landing.
SF is the *hardware motor kill* — it must be up to allow spin, and returned up after
landing as the final safety confirmation before anyone approaches the aircraft.

| # | Check | Pass/Fail |
|---|-------|-----------|
| 6.8 | SB confirmed full up before arming | ☐ |
| 6.9 | SF confirmed full up before arming | ☐ |
| 6.10 | SF moved down immediately after propellers begin spinning | ☐ |
| 6.11 | SB moved full down to initiate safe landing when required | ☐ |
| 6.12 | SF returned to full up after landing, before approaching aircraft | ☐ |

---

## Section 7 — Field Setup Verification

| # | Check | Pass/Fail |
|---|-------|-----------|
| 7.1 | Field boundary measured and marked (tape/cones) | ☐ |
| 7.2 | Aircraft positioned at exact southwest corner, nose north | ☐ |
| 7.3 | ArUco marker placed and visible within field boundary | ☐ |

**Section 7 Result:** ☐ GO ☐ NO-GO

---

## Final Authorization

All sections must read **GO** before flight is authorized.

**Cleared for flight:** ☐ Yes ☐ No
