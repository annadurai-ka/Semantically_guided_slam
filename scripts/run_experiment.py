import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from utils.config_parser import read_config
from utils.data_loader import load_image_paths, load_groundtruth
from utils.frame import Frame
from utils.matcher import match_frames
from utils.pose_estimator import estimate_pose
from utils.metrics import rotation_error, translation_error, compute_rmse
from utils.visualizer import draw_matches
import matplotlib.pyplot as plt
from utils.trajectory_logger import plot_trajectories
from tqdm import tqdm
import time



def main():
    # Change this to your dataset path
    data_dir = "data"
    img_dir = os.path.join(data_dir, "images")
    sem_dir = os.path.join(data_dir, "semantic_images")
    config_path = os.path.join(data_dir, "config.txt")
    gt_path = os.path.join(data_dir, "groundtruth.txt")

    print("Reading configuration...")
    params = read_config(config_path)
    print("Loading data...")
    img_files, sem_files = load_image_paths(img_dir, sem_dir)
    poses_gt = load_groundtruth(gt_path)
    N = len(img_files)

    if N < 16:
        print("Error: Not enough image pairs.")
        return

    # Camera intrinsics matrix
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = params.get("intirinsic_fx", 1.0)
    K[1, 1] = params.get("intirinsic_fy", 1.0)
    K[0, 2] = params.get("intirinsic_cx", 0.0)
    K[1, 2] = params.get("intirinsic_cy", 0.0)

    # ORB extractor
    extractor = cv2.ORB_create(
        nfeatures=params.get("numberOfFeatures", 1000),
        scaleFactor=params.get("fScaleFactor", 1.2),
        nlevels=int(params.get("nLevels", 8)),
        fastThreshold=int(params.get("iniThFAST", 20))
    )
    poses_gt_trajectory = []
    poses_semantic = []
    poses_normal = []


    intervals = [5, 10, 15]
    num_refs = min(250, N - max(intervals) - 2)
    ref_indices = np.floor(np.linspace(0, N - max(intervals) - 2, num_refs)).astype(int)

    rot_err_sem, trans_err_sem = [], []
    rot_err_norm, trans_err_norm = [], []
    success_sem, success_norm = 0, 0
    start_time = time.time()
    print("Running experiments...")

    for idx in tqdm(ref_indices, desc="🔁 Processing", ncols=80):
        print(f"\n→ Matching frame {idx} with offsets [5,10,15]...", end="\r")
        img1 = cv2.imread(img_files[idx], cv2.IMREAD_GRAYSCALE)
        sem1 = cv2.imread(sem_files[idx], cv2.IMREAD_GRAYSCALE)
        frame1 = Frame(img1, sem1, extractor, K=K)

        P1 = poses_gt[idx]
        T1 = np.vstack((P1, [0, 0, 0, 1]))

        for offset in intervals:
            j = idx + offset
            if j >= N:
                continue

            img2 = cv2.imread(img_files[j], cv2.IMREAD_GRAYSCALE)
            sem2 = cv2.imread(sem_files[j], cv2.IMREAD_GRAYSCALE)
            frame2 = Frame(img2, sem2, extractor, K=K)

            P2 = poses_gt[j]
            T2 = np.vstack((P2, [0, 0, 0, 1]))
            T21 = np.linalg.inv(T2) @ T1
            R_gt = T21[0:3, 0:3]
            t_gt = T21[0:3, 3].reshape(3, 1)

            for mode in ['semantic', 'normal']:
                use_sem = (mode == 'semantic')
                matches = match_frames(frame1, frame2, use_semantic=use_sem,
                                       max_dist_thresh=params.get("featureDistThreshold", 50))

                if matches.count(-1) < len(matches) - 10:
                    success, R_est, t_est = estimate_pose(frame1, frame2, matches, K)
                    if success:
                        r_err = rotation_error(R_est, R_gt)
                        t_err = translation_error(t_est, t_gt)

                        prefix = f"{idx}_{j}_{mode}"
                        out_path = os.path.join("results", "vis", f"match_{prefix}.png")
                        draw_matches(frame1, frame2, matches, out_path, title=f"{mode.upper()} MATCH")

                        if use_sem:
                            rot_err_sem.append(r_err)
                            trans_err_sem.append(t_err)
                            success_sem += 1

                            # Rebuild semantic trajectory using T21
                            T_est = np.eye(4)
                            T_est[0:3, 0:3] = R_est
                            T_est[0:3, 3] = t_est.flatten()
                            poses_semantic.append(T_est)
                        else:
                            rot_err_norm.append(r_err)
                            trans_err_norm.append(t_err)
                            success_norm += 1

                            T_est = np.eye(4)
                            T_est[0:3, 0:3] = R_est
                            T_est[0:3, 3] = t_est.flatten()
                            poses_normal.append(T_est)

                        # Ground truth pose (used for both)
                        poses_gt_trajectory.append(T21)


    elapsed = time.time() - start_time
    print(f"\n✅ Completed in {elapsed:.2f} seconds ({elapsed/60:.2f} min)")
    print("\n--- Evaluation Results ---")
    print(f"Semantic Matching:   {success_sem} successes")
    print(f"  RMSE Rotation:     {compute_rmse(rot_err_sem):.4f}°")
    print(f"  RMSE Translation:  {compute_rmse(trans_err_sem):.4f}°")
    print(f"Normal Matching:     {success_norm} successes")
    print(f"  RMSE Rotation:     {compute_rmse(rot_err_norm):.4f}°")
    print(f"  RMSE Translation:  {compute_rmse(trans_err_norm):.4f}°")
    import matplotlib.pyplot as plt

    # Plot RMSE comparison
    labels = ["Rotation RMSE (°)", "Translation RMSE (°)"]
    semantic = [compute_rmse(rot_err_sem), compute_rmse(trans_err_sem)]
    normal = [compute_rmse(rot_err_norm), compute_rmse(trans_err_norm)]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, semantic, width, label='Semantic')
    rects2 = ax.bar(x + width/2, normal, width, label='Normal')

    ax.set_ylabel('Error (°)')
    ax.set_title('RMSE Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/rmse_plot.png")
    print("📊 RMSE plot saved to results/plots/rmse_plot.png")

        # Plot trajectories
    plot_trajectories(
        poses_gt_trajectory,
        est_poses={
            "Semantic": poses_semantic,
            "Normal": poses_normal
        },
        labels=["GT", "Semantic", "Normal"],
        colors=["black", "blue", "red"],
        output_path="results/plots/trajectory_plot.png"
    )
    print("🧭 Trajectory plot saved to results/plots/trajectory_plot.png")




if __name__ == "__main__":
    main()
