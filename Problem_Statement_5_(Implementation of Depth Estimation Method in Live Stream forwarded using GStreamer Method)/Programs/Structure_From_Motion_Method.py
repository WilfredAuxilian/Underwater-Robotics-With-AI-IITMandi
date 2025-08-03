import cv2
import numpy as np

# Intrinsic camera matrix (needs calibration in real use)
K = np.array([[800, 0, 320],
              [0, 800, 240],
              [0, 0, 1]], dtype=np.float32)

def extract_features(img):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors

def match_features(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return sorted(matches, key=lambda x: x.distance)

def estimate_motion(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    return R, t, mask

def triangulate_points(pts1, pts2, K, R, t):
    P1 = np.hstack((K, np.zeros((3, 1))))
    P2 = np.hstack((K @ R, K @ t))

    pts1 = np.array(pts1).T
    pts2 = np.array(pts2).T

    points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    return (points_4d[:3] / points_4d[3]).T

def main():
    # ✅ UDP video stream from Windows via GStreamer
    gst_pipeline = (
        'udpsrc port=5600 caps="application/x-rtp, encoding-name=H264, payload=96" ! '
        'rtph264depay ! avdec_h264 ! videoconvert ! appsink drop=1'
    )

    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("❌ Error: Cannot open UDP stream.")
        return

    prev_img, prev_kp, prev_des = None, None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⛔ Frame grab failed.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = extract_features(gray)

        if prev_img is not None and prev_des is not None:
            matches = match_features(prev_des, des)
            if len(matches) > 10:
                pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

                R, t, mask = estimate_motion(pts1, pts2, K)
                mask = mask.ravel().astype(bool)

                if np.sum(mask) > 0:
                    pts1_f = pts1[mask]
                    pts2_f = pts2[mask]
                    points_3d = triangulate_points(pts1_f, pts2_f, K, R, t)

                    depths = points_3d[:, 2]
                    valid_depths = depths[depths > 0]
                    if len(valid_depths) > 0:
                        avg_depth = np.mean(valid_depths)
                        print(f"📏 Average depth: {avg_depth:.2f} meters")

                    img_matches = cv2.drawMatches(prev_img, prev_kp, gray, kp, matches[:50], None,
                                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    cv2.imshow("Feature Matches", img_matches)

        prev_img, prev_kp, prev_des = gray.copy(), kp, des

        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
