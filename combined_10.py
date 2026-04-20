import cv2
import numpy as np
import time
from adafruit_servokit import ServoKit
from picamera2 import Picamera2

# Servo Setup
kit = ServoKit(channels=16)

FEEDER_SERVO  = 0
SORT_SERVO_8  = 8    # Stage 1
SORT_SERVO_4  = 4    # Stage 2
SORT_SERVO_12 = 12   # Stage 2

FEED_PUSH = 180
FEED_REST = 0
kit.servo[FEEDER_SERVO].actuation_range = 180

STAGE1_ANGLE = {
    "Purple": 20,
    "Blue":   20,
    "Yellow": 130,
    "Red":    130,
}

STAGE2_SERVO = {
    "Purple": SORT_SERVO_4,
    "Blue":   SORT_SERVO_4,
    "Yellow": SORT_SERVO_12,
    "Red":    SORT_SERVO_12,
}

STAGE2_ANGLE = {
    "Purple": 20,
    "Blue":   130,
    "Yellow": 20,
    "Red":    130,
}

DISPLAY_COLORS = {
    "Red":    (0,   0,   255),
    "Yellow": (0,   255, 255),
    "Blue":   (255, 120, 0),
    "Purple": (255, 0,   200),
    "None":   (200, 200, 200),
}

# Camera Setup
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "BGR888"})
picam2.configure(config)
picam2.start()

# Timing
COOLDOWN   = 3.5   # Seconds between sort cycles
HOLD_TIME  = 0.5   # Seconds preventing false detections from shadow

last_action_time  = 0.0
gates_reset       = True

# Colour hold state
hold_color        = "None"
hold_start        = 0.0

# Functions

def slow_servo_move(channel, start_angle, end_angle, step_delay=0.01):
    """Step a servo smoothly from start to end angle."""
    step = 1 if end_angle > start_angle else -1
    for angle in range(int(start_angle), int(end_angle) + step, step):
        kit.servo[channel].angle = angle
        time.sleep(step_delay)
    kit.servo[channel].angle = end_angle  # Guarantee exact final position

def push_candy():
    """Slowly push candy then return feeder to rest."""
    print("Feeding...")
    slow_servo_move(FEEDER_SERVO, FEED_REST, FEED_PUSH, step_delay=0.005)
    time.sleep(0.5)
    slow_servo_move(FEEDER_SERVO, FEED_PUSH, FEED_REST, step_delay=0.005)

def cascade_sort(color):
    """Set Stage 1 and Stage 2 gates for the detected colour."""
    print(f"Sorting to {color}")
    kit.servo[SORT_SERVO_8].angle        = STAGE1_ANGLE[color]
    kit.servo[STAGE2_SERVO[color]].angle = STAGE2_ANGLE[color]
    time.sleep(0.5)  # Let gates reach position before ball arrives

def reset_gates():
    kit.servo[SORT_SERVO_8].angle  = 55
    kit.servo[SORT_SERVO_4].angle  = 55
    kit.servo[SORT_SERVO_12].angle = 55

# Start with gates in neutral
reset_gates()

# Main Loop
try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Picamera2 outputs RGB, fix for OpenCV
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # HSV masks
        masks = {
            "Red": cv2.bitwise_or(
                cv2.inRange(hsv, np.array([0,   120, 70]), np.array([10,  255, 255])),
                cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
            ),
            "Yellow": cv2.inRange(hsv, np.array([15,  100, 100]), np.array([35,  255, 255])),
            "Blue":   cv2.inRange(hsv, np.array([90,   50,  70]), np.array([130, 255, 255])),
            "Purple": cv2.inRange(hsv, np.array([130,  50,  70]), np.array([160, 255, 255])),
        }

        detected_color = "None"
        largest_area   = 0
        best_contour   = None

        for color, mask in masks.items():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 3000 and area > largest_area:
                    largest_area   = area
                    detected_color = color
                    best_contour   = cnt

        # Colour hold logic
        now = time.time()

        if detected_color != hold_color:
            # Colour changed — restart the hold timer
            hold_color = detected_color
            hold_start = now

        # Only act once the colour has been stable for HOLD_TIME seconds
        color_confirmed = (detected_color != "None") and (now - hold_start >= HOLD_TIME)

        # Annotate display
        display   = frame.copy()
        box_color = DISPLAY_COLORS.get(detected_color, (200, 200, 200))

        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            cv2.rectangle(display, (x, y), (x + w, y + h), box_color, 2)

        # Hold progress bar — fills as the colour is held steady
        hold_progress = min(1.0, (now - hold_start) / HOLD_TIME) if detected_color != "None" else 0
        bar_w = int(260 * hold_progress)
        cv2.rectangle(display, (0, 36), (260, 56), (40, 40, 40), -1)
        if bar_w > 0:
            cv2.rectangle(display, (0, 36), (bar_w, 56), box_color, -1)

        status = "CONFIRMED" if color_confirmed else detected_color
        cv2.rectangle(display, (0, 0), (260, 36), (0, 0, 0), -1)
        cv2.putText(display, f"Detected: {status}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        cv2.imshow("Candy Sorter", display)

        # Action logic
        if color_confirmed and (now - last_action_time > COOLDOWN):
            gates_reset = False
            cascade_sort(detected_color)
            push_candy()
            last_action_time = time.time()
            # Reset hold so the same ball does not trigger twice
            hold_color = "None"
            hold_start = 0.0

        elif detected_color == "None" and not gates_reset and (now - last_action_time > COOLDOWN):
            reset_gates()
            gates_reset = True

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

finally:
    reset_gates()
    kit.servo[FEEDER_SERVO].angle = FEED_REST
    picam2.stop()
    cv2.destroyAllWindows()
