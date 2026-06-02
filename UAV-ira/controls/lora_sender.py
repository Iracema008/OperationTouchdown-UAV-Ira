''' UGV communication over LoRa serial link, sends commands to UGV LoRa bridge node '''

import json
import time
import serial

from core.log import get_logger

logger = get_logger(__name__)


def _send(msg: str, cfg) -> bool:
    '''Open the serial port, write one message w \n, then close'''
    try:
        with serial.Serial(cfg.comms.serial_port, cfg.comms.baud_rate, timeout=1) as port:
            # time.sleep(0.5)
            port.write((msg + "\n").encode("ascii"))
        logger.info(f"[UGV] Sent: {msg}")
        return True

    except Exception as e:
        logger.error(f"[UGV] Failed to send '{msg}': {e}")
        return False


def send_goto(north: float, east: float, cfg) -> bool:
    '''Tell UGV to navigate to a position in the UAV NED frame '''
    # Sends JSON: {"x": <north>, "y": <east>},
    # Got rid of cmd key, bridge reads x and y directly
    msg = json.dumps({"x": round(north, 2), "y": round(east, 2)})

    return _send(msg, cfg)


def send_drive_c1(cfg) -> bool:
    '''Tell UGV to drive straight in a line for Challenge 1 '''
    return _send("STRAIGHT", cfg)


def send_stop(cfg) -> bool:
    '''Tell UGV to stop moving immediately'''
    return _send("STOP", cfg)