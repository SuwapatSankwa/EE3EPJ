import cv2                  # OpenCV library for computer vision
import numpy as np

# Open camera (0 = default camera)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("ERROR: Camera not opened")
    exit()

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Store previous detected color
prev_color = "None"

# Define Region Of Interest (ROI)
roi_x1, roi_y1 = 200, 150
roi_x2, roi_y2 = 440, 330

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame from RGB colour space to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Define HSV Colour Ranges
    red_lower = np.array([0, 120, 70])
    red_upper = np.array([10, 255, 255])

    green_lower = np.array([36, 50, 70])
    green_upper = np.array([89, 255, 255])

    dblue_lower = np.array([111, 50, 70])
    dblue_upper = np.array([130, 255, 255])

    lblue_lower = np.array([90, 50, 70])
    lblue_upper = np.array([110, 255, 255])

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    orange_lower = np.array([10, 100, 100])
    orange_upper = np.array([20, 255, 255])

    purple_lower = np.array([129, 50, 70])
    purple_upper = np.array([158, 255, 255])

    pink_lower = np.array([159, 50, 70])
    pink_upper = np.array([179, 255, 255])

    # Create Masks
    masks = {
        "Red": cv2.inRange(hsv, red_lower, red_upper),
        "Green": cv2.inRange(hsv, green_lower, green_upper),
        "Dark Blue": cv2.inRange(hsv, dblue_lower, dblue_upper),
        "Light Blue": cv2.inRange(hsv, lblue_lower, lblue_upper),
        "Yellow": cv2.inRange(hsv, yellow_lower, yellow_upper),
        "Orange": cv2.inRange(hsv, orange_lower, orange_upper),
        "Purple": cv2.inRange(hsv, purple_lower, purple_upper),
        "Pink": cv2.inRange(hsv, pink_lower, pink_upper)
    }

    detected_color = "None"
    largest_area = 0
    best_box = None

    # Find Largest Object
    for color, mask in masks.items():
        # Find shapes in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if area > 3000 and area > largest_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Check if inside detection zone
                if x > roi_x1 and x+w < roi_x2 and y > roi_y1 and y+h < roi_y2:
                    largest_area = area
                    detected_color = color
                    best_box = (x, y, w, h)
    
    # Draw Result
    if best_box is not None:
        x, y, w, h = best_box
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame, detected_color,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

    # Draw ROI Box
    cv2.rectangle(frame,(roi_x1,roi_y1),(roi_x2,roi_y2),(255,0,0),2)

    # Print only when color changes
    if detected_color != prev_color:
        print("Detected:", detected_color)
        prev_color = detected_color
    
    # Show live video feed
    cv2.imshow("Frame", frame)

    # Press 'x' to exit program
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()              # Release camera
cv2.destroyAllWindows()    # Close all windows
exit()