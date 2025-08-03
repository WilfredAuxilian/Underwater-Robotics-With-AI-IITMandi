import cv2
import numpy as np

def preprocess_underwater_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def estimate_depth_from_intensity(image):
    b, g, r = cv2.split(image)
    blue = b.astype(np.float32) + 1e-6
    red = r.astype(np.float32) + 1e-6
    ratio = blue / red
    depth_map = cv2.normalize(ratio, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_indicator = np.mean(ratio)
    return depth_map, depth_indicator

def main():
    # ✅ UDP video input via GStreamer pipeline
    gst_pipeline = (
        'udpsrc port=5600 caps="application/x-rtp, encoding-name=H264, payload=96" ! '
        'rtph264depay ! avdec_h264 ! videoconvert ! appsink drop=1'
    )

    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("❌ Error: Could not open UDP stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⛔ Frame grab failed.")
            break

        frame = preprocess_underwater_image(frame)
        depth_map, depth_indicator = estimate_depth_from_intensity(frame)
        print(f"📏 Relative depth: {depth_indicator:.2f} (higher = farther)")

        cv2.imshow("Underwater Live Feed", frame)
        cv2.imshow("Relative Depth Map", depth_map)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
