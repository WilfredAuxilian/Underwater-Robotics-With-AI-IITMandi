import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image

def load_model():
    model_id = "LiheYoung/depth-anything-large-hf"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, processor, device

def estimate_depth(model, processor, device, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb).resize((384, 384))
    inputs = processor(images=pil_image, return_tensors="pt").to(device)

    with torch.no_grad():
        depth = model(**inputs).predicted_depth.squeeze().cpu().numpy()

    return depth

def normalize_depth(depth, shape):
    depth_min, depth_max = depth.min(), depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-6)
    depth_display = cv2.resize((depth_norm * 255).astype(np.uint8), shape)
    depth_color = cv2.applyColorMap(depth_display, cv2.COLORMAP_INFERNO)
    return depth_color, depth

def get_gstreamer_pipeline():
    return (
        "udpsrc port=5600 ! "
        "application/x-rtp, encoding-name=JPEG,payload=26 ! "
        "rtpjpegdepay ! jpegdec ! "
        "videoconvert ! appsink"
    )

def main():
    model, processor, device = load_model()

    gst_pipeline = get_gstreamer_pipeline()
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("❌ Failed to open GStreamer stream.")
        return

    KNOWN_DISTANCE_METERS = 1.5
    REFERENCE_PIXEL_DEPTH = None
    SCALE_FACTOR = None

    print("📡 Streaming + Depth Anything started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame.")
            break

        depth = estimate_depth(model, processor, device, frame)
        depth_color, raw_depth = normalize_depth(depth, (frame.shape[1], frame.shape[0]))

        raw_depth_resized = cv2.resize(raw_depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
        center_depth_value = raw_depth_resized[center_y, center_x]

        if REFERENCE_PIXEL_DEPTH is None and center_depth_value > 0:
            REFERENCE_PIXEL_DEPTH = center_depth_value
            SCALE_FACTOR = KNOWN_DISTANCE_METERS / REFERENCE_PIXEL_DEPTH
            print(f"[Calibration] Scale factor set: {SCALE_FACTOR:.4f} m/relative_unit")

        if REFERENCE_PIXEL_DEPTH:
            center_depth_meters = center_depth_value * SCALE_FACTOR
            text = f"Depth @ center: {center_depth_meters:.2f} m"
        else:
            text = "Calibrating..."

        cv2.circle(depth_color, (center_x, center_y), 5, (0, 255, 0), -1)
        cv2.putText(depth_color, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        combined = np.hstack((frame, depth_color))
        cv2.imshow("GStreamer Stream | Depth Anything v2 + Meters", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
