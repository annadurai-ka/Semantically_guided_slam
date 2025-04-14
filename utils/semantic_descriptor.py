import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

def compute_semantic_histogram(segmentation, keypoint, num_classes):
    # Get keypoint center coordinates (round down to nearest integer pixel)
    x = int(keypoint.pt[0])
    y = int(keypoint.pt[1])
    # Define radius as the keypoint size (feature scale). 
    # If keypoint.size is diameter, using it directly as radius is as per the paper's guidance.
    radius = int(round(keypoint.size))
    if radius < 1:
        radius = 1  # ensure at least a 1-pixel radius

    # Image dimensions
    H, W = segmentation.shape[:2]

    # Bounding box of the window (clamped to image boundaries)
    x_min = max(0, x - radius)
    x_max = min(W - 1, x + radius)
    y_min = max(0, y - radius)
    y_max = min(H - 1, y + radius)

    # Extract the segmentation patch for the bounding box
    seg_patch = segmentation[y_min:y_max+1, x_min:x_max+1]
    h_patch, w_patch = seg_patch.shape[:2]

    # Coordinates of the keypoint relative to the patch
    center_y = y - y_min
    center_x = x - x_min

    # Create a mask for the circular region within this patch
    Y_coords, X_coords = np.ogrid[0:h_patch, 0:w_patch]  # coordinate grid for the patch
    dist_sq = (Y_coords - center_y)**2 + (X_coords - center_x)**2
    mask = dist_sq <= (radius ** 2)

    # Apply mask to get labels within the circle
    region_labels = seg_patch[mask]

    # Compute histogram of class labels in the region
    # Use np.bincount for efficiency. Ensure labels are non-negative integers.
    if region_labels.size == 0:
        # No pixels in region (edge case), return zero histogram
        hist_counts = np.zeros(num_classes, dtype=int)
    else:
        hist_counts = np.bincount(region_labels.astype(int), minlength=num_classes)
        # If segmentation labels go beyond num_classes-1, np.bincount might extend length; trim if needed
        if hist_counts.size > num_classes:
            hist_counts = hist_counts[:num_classes]
    return hist_counts

def normalize_histogram(hist_counts):
    total = hist_counts.sum()
    if total == 0:
        # Avoid division by zero; return a zero vector (or could return equal distribution)
        return np.zeros_like(hist_counts, dtype=float)
    hist_norm = hist_counts.astype(float) / float(total)
    return hist_norm

def binarize_histogram(hist_norm, threshold=0.2):
    # Create a binary mask where class proportion >= t are marked as 1
    binary_descriptor = (hist_norm >= threshold).astype(np.uint8)
    return binary_descriptor

def compute_semantic_descriptors(segmentation, keypoints, num_classes, threshold=0.2):
    descriptors = []
    for kp in keypoints:
        # 1. Compute raw semantic histogram
        hist = compute_semantic_histogram(segmentation, kp, num_classes)
        # 2. Normalize the histogram to get a distribution
        hist_norm = normalize_histogram(hist)
        # 3. Binarize the normalized histogram using threshold t
        binary_desc = binarize_histogram(hist_norm, threshold)
        descriptors.append(binary_desc)
    # Convert list to array for convenience (dtype=uint8 for binary values 0/1)
    if descriptors:
        descriptors = np.stack(descriptors, axis=0).astype(np.uint8)
    else:
        descriptors = np.empty((0, num_classes), dtype=np.uint8)
    return descriptors

