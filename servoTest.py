from adafruit_servokit import ServoKit
import time

kit = ServoKit(channels=16)

SERVO_CHANNEL = 0

while True:
    print("Move to 0°")
    kit.servo[SERVO_CHANNEL].angle = 0
    time.sleep(1)

    print("Move to 60°")
    kit.servo[SERVO_CHANNEL].angle = 60
    time.sleep(1)