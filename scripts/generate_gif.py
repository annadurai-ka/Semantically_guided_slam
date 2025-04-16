import imageio
import os
import glob

def generate_gif(input_folder, output_gif, fps=2, resize=None):
    image_files = sorted(glob.glob(os.path.join(input_folder, "*.png")))

    if not image_files:
        print("❌ No PNGs found in", input_folder)
        return

    images = []
    for img_path in image_files:
        img = imageio.imread(img_path)
        if resize:
            import cv2
            img = cv2.resize(img, resize)
        images.append(img)

    duration = 1 / fps
    imageio.mimsave(output_gif, images, duration=duration)
    print(f"🌀 GIF saved to {output_gif}")

if __name__ == "__main__":
    generate_gif("results/vis", "results/plots/slam_replay.gif", fps=2)
