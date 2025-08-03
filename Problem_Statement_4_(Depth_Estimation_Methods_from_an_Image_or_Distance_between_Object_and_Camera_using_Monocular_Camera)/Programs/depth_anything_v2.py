import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image

# Load model
model_id = "LiheYoung/depth-anything-large-hf"
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForDepthEstimation.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# Set known reference: distance to object in meters at center of frame
KNOWN_DISTANCE_METERS = 1.5  # 🔁 change this based on a known reference
REFERENCE_PIXEL_DEPTH = None

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb).resize((384, 384))
    inputs = processor(images=pil_image, return_tensors="pt").to(device)

    with torch.no_grad():
        depth = model(**inputs).predicted_depth.squeeze().cpu().numpy()

    # Normalize for display
    depth_min, depth_max = depth.min(), depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-6)
    depth_display = cv2.resize((depth_norm * 255).astype(np.uint8), (frame.shape[1], frame.shape[0]))
    depth_color = cv2.applyColorMap(depth_display, cv2.COLORMAP_INFERNO)

    # Resize depth for metric calculation
    raw_depth_resized = cv2.resize(depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Get depth at center
    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
    center_depth_value = raw_depth_resized[center_y, center_x]

    # Step 1: Capture reference once (first frame)
    if REFERENCE_PIXEL_DEPTH is None and center_depth_value > 0:
        REFERENCE_PIXEL_DEPTH = center_depth_value
        SCALE_FACTOR = KNOWN_DISTANCE_METERS / REFERENCE_PIXEL_DEPTH
        print(f"[Calibration] Scale factor set: {SCALE_FACTOR:.4f} m/relative_unit")

    # Step 2: Convert to meters
    if REFERENCE_PIXEL_DEPTH:
        center_depth_meters = center_depth_value * SCALE_FACTOR
        text = f"Depth @ center: {center_depth_meters:.2f} m"
    else:
        text = "Calibrating..."

    # Show circle and text
    cv2.circle(depth_color, (center_x, center_y), 5, (0, 255, 0), -1)
    cv2.putText(depth_color, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Stack and show
    combined = np.hstack((frame, depth_color))
    cv2.imshow("Webcam | Depth Anything v2 + Meters", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
