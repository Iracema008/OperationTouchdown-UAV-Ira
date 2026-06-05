# lora_test_send.py
import serial
import json
import time

PORT = '/dev/ttyUSB0'
BAUD = 9600

ser = serial.Serial(PORT, BAUD, timeout=2)
time.sleep(0.5)
print(f'Connected on {PORT}')

messages = [
    json.dumps({'x': 2.0, 'y': 0.0}),
    'STRAIGHT',
    'STOP',
]

for msg in messages:
    ser.write((msg + '\n').encode('ascii'))
    print(f'Sent:     {msg}')
    ack = ser.readline()
    if ack:
        print(f'ACK:      {ack.decode().strip()}')
    else:
        print('ACK:      none received')
    time.sleep(1.0)

ser.close()
print('Done')