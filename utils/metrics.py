import numpy as np
import math

def rotation_error(R_est, R_gt):
    """
    Calculate angular difference between two rotation matrices in degrees.
    
    Args:
        R_est: Estimated rotation (3x3)
        R_gt: Ground truth rotation (3x3)

    Returns:
        Rotation error in degrees
    """
    R_err = R_est @ R_gt.T
    trace = np.trace(R_err)
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)


def translation_error(t_est, t_gt):
    """
    Calculate angular error between estimated and ground truth translation vectors.

    Args:
        t_est: Estimated translation vector (3x1 or 1D array)
        t_gt: Ground truth translation vector (3x1 or 1D array)

    Returns:
        Angle between vectors in degrees
    """
    t_est = t_est.flatten()
    t_gt = t_gt.flatten()

    if np.linalg.norm(t_est) == 0 or np.linalg.norm(t_gt) == 0:
        return 0.0

    t_est /= np.linalg.norm(t_est)
    t_gt /= np.linalg.norm(t_gt)

    dot = np.dot(t_est, t_gt)
    dot = np.clip(dot, -1.0, 1.0)
    return math.degrees(math.acos(dot))


def compute_rmse(errors):
    """
    Compute Root Mean Square Error from a list of errors.

    Args:
        errors: list of float values

    Returns:
        RMSE as float
    """
    if len(errors) == 0:
        return 0.0
    errors = np.array(errors, dtype=np.float64)
    return np.sqrt(np.mean(errors**2))
