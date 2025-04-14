# scripts/orb_feature_extraction.py
import cv2
import numpy as np

def extract_orb_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    return keypoints, descriptors

def label_features(keypoints, segmentation_mask):
    labeled_features = []

    height, width = segmentation_mask.shape[:2]  # Get mask dimensions
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])

        # Ensure coordinates are within bounds
        if 0 <= x < width and 0 <= y < height:
            label1 = segmentation_mask[y, x]
            neighbors = [segmentation_mask[max(0, y-1), x], segmentation_mask[min(height-1, y+1), x],
                         segmentation_mask[y, max(0, x-1)], segmentation_mask[y, min(width-1, x+1)]] # Access mask using (row, col) format
            label = [label1] + neighbors
            
            labeled_features.append((kp, label))
        else:
            print(f"Warning: Keypoint at ({x}, {y}) is out of segmentation mask bounds.")

    return labeled_features


