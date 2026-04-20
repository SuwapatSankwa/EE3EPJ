import cv2
import numpy as np
import time
from adafruit_servokit import ServoKit
from picamera2 import Picamera2

# ─── Servo Setup ─────────────────────────────────────────────────────────────
kit = ServoKit(channels=16)

FEEDER_SERVO  = 0
SORT_SERVO_8  = 8   # Stage 1
SORT_SERVO_4  = 4   # Stage 2 Left
SORT_SERVO_12 = 12  # Stage 2 Right

FEED_PUSH = 180  
FEED_REST = 0    
kit.servo[FEEDER_SERVO].actuation_range = 180

# ─── Mapping Tables ──────────────────────────────────────────────────────────
STAGE1_ANGLE = {
    "Blue": 45, "Purple": 45, "Red": 135, "Yellow": 135
}

STAGE2_SERVO = {
    "Blue": SORT_SERVO_4, "Purple": SORT_SERVO_4,
    "Red": SORT_SERVO_12, "Yellow": SORT_SERVO_12
}

STAGE2_ANGLE = {
    "Blue": 135, "Purple": 30, "Red": 135, "Yellow": 30
}

DISPLAY_COLORS = {
    "Red": (0, 0, 255), "Yellow": (0, 255, 255),
    "Blue": (255, 0, 0), "Purple": (255, 0, 200),
    "None": (255, 255, 255),
}

# ─── Camera Setup ────────────────────────────────────────────────────────────
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "BGR888"})
picam2.configure(config)
picam2.start()

# ─── Timing & Cooldown ───────────────────────────────────────────────────────
last_action_time = 0
COOLDOWN = 3.5  # Increased to allow the ball to roll through the whole system

# ─── Functions ───────────────────────────────────────────────────────────────

def slow_servo_move(servo_channel, start_angle, end_angle, step_delay=0.01):
    """Moves a servo slowly by stepping through angles."""
    step = 1 if end_angle > start_angle else -1
    for angle in range(int(start_angle), int(end_angle) + step, step):
        kit.servo[servo_channel].angle = angle
        time.sleep(step_delay)

def push_candy():
    """Slowly pushes the candy and returns to rest."""
    print("Feeding...")
    # Slow push
    slow_servo_move(FEEDER_SERVO, FEED_REST, FEED_PUSH, step_delay=0.005)
    time.sleep(0.5) 
    # Slow return
    slow_servo_move(FEEDER_SERVO, FEED_PUSH, FEED_REST, step_delay=0.005)

def cascade_sort(color):
    """Sets the sorting gates."""
    kit.servo[SORT_SERVO_8].angle = STAGE1_ANGLE[color]
    kit.servo[STAGE2_SERVO[color]].angle = STAGE2_ANGLE[color]
    time.sleep(0.5) # Time for gates to physically reach positions

# ─── Main Loop ───────────────────────────────────────────────────────────────
try:
    while True:
        frame = picam2.capture_array()
        # Removed the flip line [:, :, ::-1] as it breaks OpenCV's BGR expectations
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # HSV Masks
        masks = {
            "Red": cv2.bitwise_or(
                cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])),
                cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
            ),
            "Yellow": cv2.inRange(hsv, np.array([15, 100, 100]), np.array([35, 255, 255])),
            "Blue":   cv2.inRange(hsv, np.array([90, 50, 70]),   np.array([130, 255, 255])),
            "Purple": cv2.inRange(hsv, np.array([130, 50, 70]),  np.array([160, 255, 255])),
        }

        detected_color = "None"
        largest_area = 0
        best_contour = None

        for color, mask in masks.items():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 3000 and area > largest_area:
                    largest_area = area
                    detected_color = color
                    best_contour = cnt

        # Annotate
        display = frame.copy()
        box_color = DISPLAY_COLORS.get(detected_color, (255, 255, 255))
        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            cv2.rectangle(display, (x, y), (x + w, y + h), box_color, 2)
        
        cv2.putText(display, f"Detected: {detected_color}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
        cv2.imshow("Candy Sorter", display)

        # Action Logic
        current_time = time.time()
        if detected_color != "None" and (current_time - last_action_time > COOLDOWN):
            # Move gates FIRST, then push
            cascade_sort(detected_color)
            push_candy()
            last_action_time = time.time()
        
        elif detected_color == "None" and (current_time - last_action_time > COOLDOWN):
            # Reset gates to neutral while waiting
            kit.servo[SORT_SERVO_8].angle  = 90
            kit.servo[SORT_SERVO_4].angle  = 90
            kit.servo[SORT_SERVO_12].angle = 90

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

finally:
    kit.servo[FEEDER_SERVO].angle = FEED_REST
    picam2.stop()
    cv2.destroyAllWindows()