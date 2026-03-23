import cv2
import numpy as np
import time
from adafruit_servokit import ServoKit
from picamera2 import Picamera2

# Servo Setup
kit = ServoKit(channels=16)

FEEDER_SERVO = 0
SORT_SERVO = 1

PUSH = 50
REST = 0

# Servo angles for each colour
COLOR_ANGLES = {
    "Red": 20,
    "Yellow": 60,
    "Blue": 100,
    "Purple": 140
}

# Camera Setup (IMPORTANT FIX)
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "BGR888"}
)
picam2.configure(config)
picam2.start()

# Cooldown Setup
last_action_time = 0
COOLDOWN = 1.5  # seconds

# Functions
def push_candy():
    kit.servo[FEEDER_SERVO].angle = PUSH
    time.sleep(0.4)
    kit.servo[FEEDER_SERVO].angle = REST

def sort_to(angle):
    kit.servo[SORT_SERVO].angle = angle
    time.sleep(0.3)

# Main Loop
while True:
    frame = picam2.capture_array()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HSV Colour Ranges
    masks = {
        "Red": cv2.inRange(hsv, np.array([0,120,70]), np.array([10,255,255])),
        "Yellow": cv2.inRange(hsv, np.array([20,100,100]), np.array([30,255,255])),
        "Blue": cv2.inRange(hsv, np.array([90,50,70]), np.array([130,255,255])),
        "Purple": cv2.inRange(hsv, np.array([130,50,70]), np.array([160,255,255]))
    }

    detected_color = "None"
    largest_area = 0

    # Detection Logic
    for color, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area > 3000 and area > largest_area:
                largest_area = area
                detected_color = color

    # Cooldown + Action
    current_time = time.time()

    if detected_color != "None" and (current_time - last_action_time > COOLDOWN):
        print("Detected:", detected_color)

        angle = COLOR_ANGLES.get(detected_color, 0)
        sort_to(angle)
        push_candy()

        last_action_time = current_time

    #cv2.imshow("Frame", frame) # remove if SSH

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cv2.destroyAllWindows()