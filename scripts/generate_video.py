import cv2
import os
import glob

def generate_video(input_folder, output_path, fps=2, resize_scale=1.0):
    image_files = sorted(glob.glob(os.path.join(input_folder, "*.png")))

    if not image_files:
        print("❌ No images found in:", input_folder)
        return

    # Read first frame for dimensions
    frame = cv2.imread(image_files[0])
    if resize_scale != 1.0:
        frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)

    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for img_path in image_files:
        img = cv2.imread(img_path)
        if resize_scale != 1.0:
            img = cv2.resize(img, (w, h))
        out.write(img)

    out.release()
    print(f"🎥 Video saved to {output_path}")

if __name__ == "__main__":
    generate_video("results/vis", "results/plots/slam_replay.mp4", fps=2, resize_scale=1.0)
