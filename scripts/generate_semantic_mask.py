import os
import cv2
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101

# Paths
image_dir = "/home/kavi/semantically_guided_slam/data/images"
output_dir = "/home/kavi/semantically_guided_slam/data/semantic_images"

os.makedirs(output_dir, exist_ok=True)

# Load pre-trained DeepLabV3 model
model = deeplabv3_resnet101(pretrained=True).eval()

# Image transform
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((480, 640)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Process each image
image_files = sorted([f for f in os.listdir(image_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

print(f"Generating semantic masks for {len(image_files)} images...")

for fname in image_files:
    path = os.path.join(image_dir, fname)
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_tensor = transform(image_rgb).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    labels = output.argmax(0).byte().cpu().numpy()

    out_path = os.path.join(output_dir, fname)
    cv2.imwrite(out_path, labels)

    print(f"Saved: {out_path}")

print("✅ Semantic masks saved to:", output_dir)
