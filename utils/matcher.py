import cv2
import numpy as np

def compare_semantics(desc1, desc2):
    """
    Compare two semantic descriptors (histograms).
    Returns Hamming-like distance (binary mismatch of class presence).
    """
    if desc1.size == 0 or desc2.size == 0:
        return max(desc1.size, desc2.size)

    len1, len2 = len(desc1), len(desc2)
    if len1 < len2:
        desc1 = np.pad(desc1, (0, len2 - len1))
    elif len2 < len1:
        desc2 = np.pad(desc2, (0, len1 - len2))

    bin1 = (desc1 > 0).astype(np.uint8)
    bin2 = (desc2 > 0).astype(np.uint8)

    return int(np.sum(np.bitwise_xor(bin1, bin2)))


def match_frames(frame1, frame2, use_semantic=False, max_dist_thresh=50, nn_ratio=0.7):
    """
    Match keypoints between two frames with or without semantic filtering.
    Args:
        frame1, frame2: Frame objects with keypoints, descriptors, semantic_descriptors
        use_semantic: bool, whether to apply semantic guidance
        max_dist_thresh: max descriptor distance threshold
        nn_ratio: Lowe's ratio test threshold
    Returns:
        matches_12: list of indices - matches_12[i] = j means frame1.keypoint[i] matched frame2.keypoint[j]
    """
    matches_12 = [-1] * len(frame1.keypoints)
    if len(frame1.keypoints) == 0 or len(frame2.keypoints) == 0:
        return matches_12

    coords2 = [kp.pt for kp in frame2.keypoints]
    levels2 = [kp.octave for kp in frame2.keypoints]

    HISTO_LENGTH = 30
    rot_hist = [[] for _ in range(HISTO_LENGTH)]
    factor = 1.0 / HISTO_LENGTH

    best_dist_for_j = [float('inf')] * len(frame2.keypoints)
    match_for_j = [-1] * len(frame2.keypoints)

    window_size = 300
    nmatches = 0

    for i1, kp1 in enumerate(frame1.keypoints):
        level1 = kp1.octave
        if level1 > 0:
            continue
        x1, y1 = kp1.pt

        # Filter candidates
        min_x, max_x = x1 - window_size, x1 + window_size
        min_y, max_y = y1 - window_size, y1 + window_size
        candidates = [j2 for j2, (pt2, lvl2) in enumerate(zip(coords2, levels2))
                      if lvl2 == level1 and min_x <= pt2[0] <= max_x and min_y <= pt2[1] <= max_y]

        if not candidates:
            continue

        d1 = frame1.descriptors[i1]
        best_dist = float('inf')
        second_best_dist = float('inf')
        best_j2 = -1

        for j2 in candidates:
            d2 = frame2.descriptors[j2]
            if d1.dtype == np.uint8:
                dist = cv2.norm(d1, d2, cv2.NORM_HAMMING)
            else:
                dist = np.linalg.norm(d1 - d2)

            if use_semantic:
                sem1 = frame1.semantic_descriptors[i1]
                sem2 = frame2.semantic_descriptors[j2]
                dist_sem = compare_semantics(sem1, sem2)
                max_desc = 256 if d1.dtype == np.uint8 else 1.414
                dist = (0.9 * dist) + (0.1 * max_desc / max(len(sem1), len(sem2)) * dist_sem)

            if dist < best_dist:
                second_best_dist = best_dist
                best_dist = dist
                best_j2 = j2
            elif dist < second_best_dist:
                second_best_dist = dist

        if best_dist <= max_dist_thresh and best_dist < second_best_dist * nn_ratio:
            if match_for_j[best_j2] != -1:
                prev_i = match_for_j[best_j2]
                matches_12[prev_i] = -1
                nmatches -= 1
            matches_12[i1] = best_j2
            match_for_j[best_j2] = i1
            best_dist_for_j[best_j2] = best_dist
            nmatches += 1

            angle_diff = kp1.angle - frame2.keypoints[best_j2].angle
            angle_diff = angle_diff + 360 if angle_diff < 0 else angle_diff
            bin_idx = int(round(angle_diff * factor)) % HISTO_LENGTH
            rot_hist[bin_idx].append(i1)

    # Orientation consistency filter
    counts = [len(h) for h in rot_hist]
    if len(counts) >= 3:
        ind1 = np.argmax(counts)
        c1 = counts[ind1]
        counts[ind1] = -1
        ind2 = np.argmax(counts)
        counts[ind2] = -1
        ind3 = np.argmax(counts)
        for k in range(HISTO_LENGTH):
            if k != ind1 and k != ind2 and k != ind3:
                for i1 in rot_hist[k]:
                    if matches_12[i1] != -1:
                        matches_12[i1] = -1
                        nmatches -= 1

    return matches_12
