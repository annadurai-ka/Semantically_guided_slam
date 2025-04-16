import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
from semantic_segmentation import load_deeplabv3, segment_image
from utils.visualization import overlay_segmentation

os.makedirs("results/semantic_samples", exist_ok=True)

model = load_deeplabv3()
image_dir = "data/images"
out_dir = "results/semantic_samples"

for fname in sorted(os.listdir(image_dir))[:5]:  # Show first 5 examples
    image_path = os.path.join(image_dir, fname)
    image = cv2.imread(image_path)
    pil_image, seg_mask = segment_image(image_path, model)
    overlay = overlay_segmentation(image, seg_mask)

    out_path = os.path.join(out_dir, f"overlay_{fname}")
    cv2.imwrite(out_path, overlay)
    print(f"✅ Saved {out_path}")
