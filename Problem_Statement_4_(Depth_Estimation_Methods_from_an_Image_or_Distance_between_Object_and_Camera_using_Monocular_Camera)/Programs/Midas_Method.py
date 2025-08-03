import cv2
import torch
import numpy as np
import time

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)                     # Loading MiDaS model
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)           # Loading transforms
transform = midas_transforms.small_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                         # Move model to device
midas.to(device)

cap = cv2.VideoCapture(0) # Start webcam
if not cap.isOpened():
    print("Cannot Open Webcam")
    exit()

print("Webcam and MiDaS initialized. Press 'q' to quit.")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                               # Convert BGR to RGB

    input_image = transform(img).to(device)                                                    # Apply MiDaS transform
    input_batch = input_image                                                                  # already batched

    with torch.no_grad():                                                                      # Depth inference
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy()

    depth_min = depth_map.min()                                                                # Normalize depth map for visualization
    depth_max = depth_map.max()
    depth_vis = (255 * (depth_map - depth_min) / (depth_max - depth_min)).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

    current_time = time.time()                                                                  # Calculate FPS
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),                                             # Draw FPS on original frame
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    h, w = depth_map.shape                                                                      # Draw depth value at center of depth map
    center_x, center_y = w // 2, h // 2
    center_depth = depth_map[center_y, center_x]
    cv2.putText(depth_colormap, f"Depth: {center_depth:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    combined = np.hstack((frame, depth_colormap))                                             
    cv2.imshow("Webcam + MiDaS Depth", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()