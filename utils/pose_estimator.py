import cv2
import numpy as np

def estimate_pose(frame1, frame2, matches, K):
    """
    Estimate relative pose (R, t) from frame1 to frame2 using matched keypoints.
    
    Args:
        frame1, frame2: Frame objects
        matches: list of indices - matches[i] = j means frame1.kp[i] matched frame2.kp[j]
        K: 3x3 camera intrinsic matrix

    Returns:
        success (bool), R (3x3 rotation matrix), t (3x1 translation vector)
    """
    pts1, pts2 = [], []

    for i1, j2 in enumerate(matches):
        if j2 != -1:
            pts1.append(frame1.keypoints[i1].pt)
            pts2.append(frame2.keypoints[j2].pt)

    if len(pts1) < 8:
        return False, None, None

    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)

    # E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=10.0)




    if E is None or E.shape != (3, 3):
        return False, None, None

    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

    return True, R, t
