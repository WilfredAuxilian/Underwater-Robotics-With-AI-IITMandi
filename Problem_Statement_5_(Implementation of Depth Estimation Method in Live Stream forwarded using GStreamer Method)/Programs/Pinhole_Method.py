import cv2
import numpy as np

# Camera parameters
KNOWN_OBJECT_WIDTH = 0.5     # in meters (real object width)
FOCAL_LENGTH = 800           # in pixels (adjust after calibration)

def calculate_distance(pixel_width):
    if pixel_width > 0:
        return (KNOWN_OBJECT_WIDTH * FOCAL_LENGTH) / pixel_width
    return None

def detect_seabed_feature(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        pixel_width = rect[1][0]  # Width of bounding box in pixels
        return pixel_width, largest_contour
    return 0, None

# GStreamer pipeline for receiving UDP stream from Windows (port 5600)
gst_pipeline = (
    'udpsrc port=5600 caps="application/x-rtp, encoding-name=H264, payload=96" ! '
    'rtph264depay ! avdec_h264 ! videoconvert ! appsink drop=1'
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ Error: Could not open UDP stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⛔ Frame not received.")
        continue

    pixel_width, contour = detect_seabed_feature(frame)
    distance = calculate_distance(pixel_width)

    if distance:
        cv2.putText(frame, f"Distance: {distance:.2f} m", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

    cv2.imshow("UDP Stream + Distance Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
