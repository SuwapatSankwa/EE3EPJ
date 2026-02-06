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

    # Create Binary Masks for Each Colour
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    dblue_mask = cv2.inRange(hsv, dblue_lower, dblue_upper)
    lblue_mask = cv2.inRange(hsv, lblue_lower, lblue_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
    purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)
    pink_mask = cv2.inRange(hsv, pink_lower, pink_upper)

    # Dictionary to store colour names and their masks
    masks = {
        "Red": red_mask,
        "Green": green_mask,
        "Dark Blue": dblue_mask,
        "Light Blue": lblue_mask,
        "Yellow": yellow_mask,
        "Orange": orange_mask,
        "Purple": purple_mask,
        "Pink": pink_mask
    }

    detected_color = "None"

    # Find Contours for Each Colour Mask
    for color, mask in masks.items():
        # Find shapes in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Ignore small noisy areas
            if area > 3000:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(cnt)
                # Draw rectangle around detected object
                cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
                # Display detected colour name
                cv2.putText(frame, color, (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 255, 0), 2)
                detected_color = color
    
    # Print detected colour in terminal
    print("Detected:", detected_color)
    
    # Show live video feed
    cv2.imshow("Frame", frame)

    # Press 'x' to exit program
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()              # Release camera
cv2.destroyAllWindows()    # Close all windows
exit()