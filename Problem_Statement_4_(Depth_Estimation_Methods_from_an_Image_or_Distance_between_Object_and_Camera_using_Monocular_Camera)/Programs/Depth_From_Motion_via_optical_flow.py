import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

ret, prev_frame = cap.read()                                                        # Read the first frame
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

print("Depth from Motion running... Press 'q' to quit.")

while True:
    start_time = time.time()

    ret, frame = cap.read()                                                         # Read the next frame
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,                              # Compute optical flow
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])                        # Flow magnitude as proxy for inverse depth (closer objects move more)

    inv_depth = 1 / (mag + 1e-5)                                                  # Normalize inverse magnitude to simulate depth
    inv_depth_normalized = cv2.normalize(inv_depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_display = inv_depth_normalized.astype(np.uint8)

    depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_INFERNO)        # Apply colormap for visualization

    fps = 1.0 / (time.time() - start_time)                                        # FPS Counter
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    h, w = mag.shape                                                              # Display example depth value at the center
    depth_val = inv_depth[int(h/2), int(w/2)]
    cv2.putText(depth_colored, f"Depth(center): {depth_val:.2f} (relative units)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    combined = np.hstack((frame, depth_colored))                                  # Combine original frame and depth map side-by-side

    cv2.imshow("Depth from Motion (Monocular)", combined)

    prev_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()