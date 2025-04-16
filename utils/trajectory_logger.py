import numpy as np
import matplotlib.pyplot as plt
import os

def plot_trajectories(gt_poses, est_poses, labels, colors, output_path):
    """
    Plot 2D top-down view (x,z) of trajectories.
    """
    plt.figure(figsize=(8, 6))

    # Ground truth
    gt_x = [T[0, 3] for T in gt_poses]
    gt_z = [T[2, 3] for T in gt_poses]
    plt.plot(gt_x, gt_z, color='black', label="GT", linewidth=2)

    # Estimated
    for name, traj in est_poses.items():
        x = [T[0, 3] for T in traj]
        z = [T[2, 3] for T in traj]
        color = colors[labels.index(name)]
        plt.plot(x, z, label=name, linestyle='--', linewidth=2, color=color)

    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("Trajectory Comparison")
    plt.legend()
    plt.axis("equal")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
