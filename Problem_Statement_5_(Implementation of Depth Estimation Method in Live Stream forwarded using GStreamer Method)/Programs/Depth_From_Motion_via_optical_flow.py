import cv2
import numpy as np
import time

# ✅ UDP input pipeline using GStreamer
gst_pipeline = (
    'udpsrc port=5600 caps="application/x-rtp, encoding-name=H264, payload=96" ! '
    'rtph264depay ! avdec_h264 ! videoconvert ! appsink drop=1'
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("❌ Cannot open UDP webcam stream.")
    exit()

ret, prev_frame = cap.read()
if not ret:
    print("❌ Could not read initial frame.")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

print("🔁 Depth from Motion running... Press 'q' to quit.")

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("⛔ Frame grab failed.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ▶️ Optical Flow calculation (Farneback)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    inv_depth = 1 / (mag + 1e-5)
    inv_depth_normalized = cv2.normalize(inv_depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_display = inv_depth_normalized.astype(np.uint8)

    depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_INFERNO)

    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    h, w = mag.shape
    center_depth = inv_depth[int(h / 2), int(w / 2)]
    cv2.putText(depth_colored, f"Depth(center): {center_depth:.2f} (rel)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    combined = np.hstack((frame, depth_colored))
    cv2.imshow("Depth from Motion (UDP Camera)", combined)

    prev_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
