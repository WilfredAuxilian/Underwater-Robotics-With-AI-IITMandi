import cv2
import numpy as np

                                                                       # Camera intrinsic parameters (calibrate your camera or use approximate values)
K = np.array([[800, 0, 320], 
              [0, 800, 240],
              [0, 0, 1]], dtype=np.float32)

def extract_features(img):                                             # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors

def match_features(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)              # Match features between two frames using BFMatcher.
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def estimate_motion(pts1, pts2, K):                                    # Estimate essential matrix and recover camera pose.
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    return R, t, mask

def triangulate_points(pts1, pts2, K, R, t):                           # Triangulate 3D points from matched 2D points.
    pts1 = np.array(pts1, dtype=np.float32).reshape(-1, 2)
    pts2 = np.array(pts2, dtype=np.float32).reshape(-1, 2)

    P1 = np.hstack((K, np.zeros((3, 1))))                               # Projection matrices
    P2 = np.hstack((K @ R, K @ t.reshape(3, 1)))

    pts1 = pts1.T                                                       # Transpose points to 2xN format for cv2.triangulatePoints
    pts2 = pts2.T  

    points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)               # Triangulate
    points_3d = points_4d[:3] / points_4d[3]                            # Convert to homogeneous coordinates
    return points_3d.T  

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    prev_img = None                                                     # Variables to store previous frame data
    prev_kp = None
    prev_des = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        kp, des = extract_features(gray)

        if prev_img is not None and prev_des is not None:
            matches = match_features(prev_des, des)                                                         # Match features with previous frame
            if len(matches) > 10: 
                pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 2)
                pts2 = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 2)

                R, t, mask = estimate_motion(pts1, pts2, K)                                                 # Estimate camera motion

                mask = mask.ravel() > 0                                                                     # Filter points using mask
                if np.sum(mask) > 0: 
                    points_3d = triangulate_points(pts1[mask], pts2[mask], K, R, t)                         # Triangulate 3D points

                    depths = points_3d[:, 2]                                                                # Compute depth (z-coordinate)
                    valid_depths = depths[depths > 0] 
                    if len(valid_depths) > 0:
                        avg_depth = np.mean(valid_depths)
                        print(f"Average depth: {avg_depth:.2f} meters")

                    img_matches = cv2.drawMatches(prev_img, prev_kp, gray, kp, matches[:50], None,          # Draw matches
                                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    cv2.imshow('Feature Matches', img_matches)

        prev_img = gray.copy()                                                                              # Update previous frame data
        prev_kp = kp
        prev_des = des

        cv2.imshow('Live Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()