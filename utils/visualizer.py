import cv2
import numpy as np
import os

def draw_matches(frame1, frame2, matches, output_path, title=""):
    """
    Draw and save matches between two frames.
    
    Args:
        frame1, frame2: Frame objects with .image and .keypoints
        matches: list of indices where matches[i] = j (or -1)
        output_path: where to save the match image
        title: optional title text
    """
    # Convert grayscale images to color for drawing
    img1 = cv2.cvtColor(frame1.image, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(frame2.image, cv2.COLOR_GRAY2BGR)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    vis = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:] = img2

    for i1, i2 in enumerate(matches):
        if i2 == -1:
            continue
        pt1 = tuple(np.round(frame1.keypoints[i1].pt).astype(int))
        pt2 = tuple(np.round(frame2.keypoints[i2].pt).astype(int))
        pt2_shifted = (int(pt2[0] + w1), int(pt2[1]))

        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(vis, pt1, pt2_shifted, color, 1, lineType=cv2.LINE_AA)
        cv2.circle(vis, pt1, 3, color, -1)
        cv2.circle(vis, pt2_shifted, 3, color, -1)

    if title:
        cv2.putText(vis, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis)
