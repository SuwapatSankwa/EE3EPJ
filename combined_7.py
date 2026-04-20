import cv2
import numpy as np
import time
from adafruit_servokit import ServoKit
from picamera2 import Picamera2

# ─── Servo Setup ─────────────────────────────────────────────────────────────
kit = ServoKit(channels=16)

FEEDER_SERVO  = 0
SORT_SERVO_8  = 8   
SORT_SERVO_4  = 4   
SORT_SERVO_12 = 12  

FEED_PUSH = 180  
FEED_REST = 0    
kit.servo[FEEDER_SERVO].actuation_range = 180

# ─── Mapping Tables ──────────────────────────────────────────────────────────
STAGE1_ANGLE = {"Blue": 45, "Purple": 45, "Red": 135, "Yellow": 135}
STAGE2_SERVO = {"Blue": SORT_SERVO_4, "Purple": SORT_SERVO_4, "Red": SORT_SERVO_12, "Yellow": SORT_SERVO_12}
STAGE2_ANGLE = {"Blue": 135, "Purple": 30, "Red": 135, "Yellow": 30}
DISPLAY_COLORS = {"Red": (0, 0, 255), "Yellow": (0, 255, 255), "Blue": (255, 0, 0), "Purple": (255, 0, 200), "None": (255, 255, 255)}

# ─── Camera Setup ────────────────────────────────────────────────────────────
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "BGR888"})
picam2.configure(config)
picam2.start()

# ─── Timing & State ──────────────────────────────────────────────────────────
last_action_time = 0
# Total time: Feeder Move (~2s) + Rolling Time (~3s) = 5s
COOLDOWN = 5.0  
is_processing = False

# ─── Functions ───────────────────────────────────────────────────────────────

def slow_servo_move(servo_channel, start_angle, end_angle, step_delay=0.012):
    """Moves a servo slowly. Higher step_delay = slower movement."""
    step = 1 if end_angle > start_angle else -1
    for angle in range(int(start_angle), int(end_angle) + step, step):
        angle = max(0, min(180, angle))
        kit.servo[servo_channel].angle = angle
        time.sleep(step_delay)

def process_one_ball(color):
    """The full physical sequence. The camera logic waits for this to finish."""
    print(f">>> STARTING SORT: {color}")
    
    # 1. Position Sorting Gates
    kit.servo[SORT_SERVO_8].angle = STAGE1_ANGLE[color]
    kit.servo[STAGE2_SERVO[color]].angle = STAGE2_ANGLE[color]
    time.sleep(0.8) # Give gates time to move
    
    # 2. Slow Feeder Push
    print("Pushing...")
    slow_servo_move(FEEDER_SERVO, FEED_REST, FEED_PUSH, step_delay=0.01)
    time.sleep(0.5) # Pause at the top
    
    # 3. Slow Feeder Return
    print("Returning...")
    slow_servo_move(FEEDER_SERVO, FEED_PUSH, FEED_REST, step_delay=0.01)
    
    # 4. Final Wait
    print("Waiting for ball to exit chutes...")
    time.sleep(2.0) # Additional time for the ball to travel through Stage 2
    print(">>> READY FOR NEXT BALL")

# ─── Main Loop ───────────────────────────────────────────────────────────────
try:
    while True:
        # Capture frame
        frame = picam2.capture_array()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect Colors
        masks = {
            "Red": cv2.bitwise_or(
                cv2.inRange(hsv, np.array([0, 150, 100]), np.array([10, 255, 255])),
                cv2.inRange(hsv, np.array([170, 150, 100]), np.array([180, 255, 255]))
            ),
            "Yellow": cv2.inRange(hsv, np.array([15, 120, 120]), np.array([35, 255, 255])),
            "Blue":   cv2.inRange(hsv, np.array([100, 150, 80]),  np.array([130, 255, 255])),
            "Purple": cv2.inRange(hsv, np.array([135, 100, 80]),  np.array([160, 255, 255])),
        }

        detected_color = "None"
        largest_area = 0
        best_contour = None

        for color, mask in masks.items():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 4500 and area > largest_area:
                    largest_area = area
                    detected_color = color
                    best_contour = cnt

        # Visual Annotations
        display = frame.copy()
        box_color = DISPLAY_COLORS.get(detected_color, (255, 255, 255))
        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            cv2.rectangle(display, (x, y), (x + w, y + h), box_color, 2)
        
        cv2.putText(display, f"Status: {'WAITING' if (time.time() - last_action_time < COOLDOWN) else 'READY'}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Detected: {detected_color}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        
        cv2.imshow("Candy Sorter", display)

        # Action Logic
        current_time = time.time()
        
        # Only act if we aren't in cooldown and we actually see a color
        if detected_color != "None" and (current_time - last_action_time > COOLDOWN):
            # This function blocks the loop until the whole physical movement is done
            process_one_ball(detected_color)
            
            # Reset gates to neutral
            kit.servo[SORT_SERVO_8].angle  = 90
            kit.servo[SORT_SERVO_4].angle  = 90
            kit.servo[SORT_SERVO_12].angle = 90
            
            # NOW start the cooldown timer
            last_action_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

finally:
    print("Closing...")
    kit.servo[FEEDER_SERVO].angle = FEED_REST
    picam2.stop()
    cv2.destroyAllWindows()