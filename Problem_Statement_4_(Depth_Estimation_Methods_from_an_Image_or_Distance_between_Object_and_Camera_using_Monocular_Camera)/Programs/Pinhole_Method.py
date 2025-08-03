import cv2
import numpy as np

                                                                                        # Camera parameters for this method
KNOWN_OBJECT_WIDTH = 0.5                                                                # a known seabed feature size, measured in meters
FOCAL_LENGTH = 800                                                                      # calibrate your camera based on pixel size, measured in pixels

def calculate_distance(pixel_width):
    if pixel_width > 0:
        distance = (KNOWN_OBJECT_WIDTH * FOCAL_LENGTH) / pixel_width
        return distance
    return None

def detect_seabed_feature(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                                      # Convertion into grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)                                         # Applying blur to reduce noise
    edges = cv2.Canny(blurred, 50, 150)                                                 # Edge detection using Canny Filter
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # Finding contours

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)                            # Getting the largest contour among other
        rect = cv2.minAreaRect(largest_contour)                                         # Get the width of the bounding rectangle
        pixel_width = rect[1][0]                                           # Width in pixels which gets converted into pixel width and returns as result
        return pixel_width
    return 0

cap = cv2.VideoCapture(0)                                                               #  Open the camera and 0 is the default camera index

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    ret, frame = cap.read()                                                             # Capture a single frame
    if ret:
        pixel_width = detect_seabed_feature(frame)                                      # Detect seabed feature
        
        distance = calculate_distance(pixel_width)                                      # pixel width goes in Calculate distance function
        
        if distance:                                                                    #  Display results
            print(f"Pixel Width: {pixel_width:.2f} pixels")
            print(f"Distance to Seabed: {distance:.2f} meters")
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                              # Draw the contour on the frame for visualization
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                cv2.putText(frame, f"Distance: {distance:.2f} m", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Camera Feed", frame)
                cv2.waitKey(0) 
        else:
            print("No valid object detected for distance calculation.")
    else:
        print("Error: Could not read frame.")
    
    cap.release()
    cv2.destroyAllWindows()