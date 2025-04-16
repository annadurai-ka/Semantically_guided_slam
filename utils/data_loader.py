import os
import cv2
import numpy as np

def load_image_paths(img_dir, sem_dir):
    """
    Load and sort image and semantic image file paths.
    Returns:
        - img_files: list of image file paths
        - sem_files: list of semantic image file paths
    """
    img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                        if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    sem_files = sorted([os.path.join(sem_dir, f) for f in os.listdir(sem_dir) 
                        if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if len(img_files) != len(sem_files):
        raise ValueError(f"Number of images ({len(img_files)}) and semantic images ({len(sem_files)}) do not match.")
    
    return img_files, sem_files

def load_groundtruth(gt_path):
    """
    Load ground truth poses from groundtruth.txt.
    Supports:
        - 8 or 9-column TUM format: timestamp x y z qx qy qz qw
        - 12 or 13-column [R|t] matrix format (row-major)
    Returns:
        - poses: numpy array of shape (N, 3, 4)
    """
    poses = []
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = list(map(float, line.split()))
            if len(vals) == 9:  # timestamp + TUM format
                vals = vals[1:]
            if len(vals) == 8:  # TUM format
                vals = vals[1:]
                tx, ty, tz, qx, qy, qz, qw = vals
                norm = np.linalg.norm([qx, qy, qz, qw])
                qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
                R = np.array([
                    [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
                    [2*(qx*qy + qz*qw),         1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                    [2*(qx*qz - qy*qw),         2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)]
                ], dtype=np.float32)
                t = np.array([[tx], [ty], [tz]], dtype=np.float32)
                pose = np.hstack((R, t))  # 3x4
                poses.append(pose)
            elif len(vals) == 13:  # timestamp + 3x4 row-major
                vals = vals[1:]
            if len(vals) == 12:  # 3x4
                pose = np.array(vals, dtype=np.float32).reshape(3, 4)
                poses.append(pose)
    return np.array(poses, dtype=np.float32)
