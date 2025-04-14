# utils/visualization.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import matplotlib.pyplot as plt

def visualize_keypoints(image, keypoints, window_name="Keypoints"):
    output_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
    cv2.imshow(window_name, output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def overlay_segmentation(image, mask, alpha=0.5):
    color_mask = cv2.applyColorMap(mask * 15, cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(image, alpha, color_mask, 1 - alpha, 0)
    return overlayed
