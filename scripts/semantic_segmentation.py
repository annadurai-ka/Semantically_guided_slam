# scripts/semantic_segmentation.py
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

def load_deeplabv3():
    model = torch.hub.load('pytorch/vision:v0.13.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval()
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    return image, input_tensor

def segment_image(image_path, model):
    image, input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    segmentation_mask = output.argmax(0).byte().cpu().numpy()
    # Resize segmentation mask back to original image dimensions
    original_image = np.array(image)
    segmentation_mask_resized = cv2.resize(segmentation_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return image, segmentation_mask_resized

