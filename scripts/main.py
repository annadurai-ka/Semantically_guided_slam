# scripts/main.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from semantic_segmentation import load_deeplabv3, segment_image
from orb_feature_extraction import extract_orb_features, label_features
from utils.visualization import visualize_keypoints, overlay_segmentation
import cv2
import os
from utils.semantic_descriptor import compute_semantic_descriptors


def main():
    try:
        # Load model and image
        print("Loading the semantic segmentation model...")
        model = load_deeplabv3()
        print("Model loaded successfully!")

        # Check if the image file exists
        image_path = "data/01.png"
        if not os.path.isfile(image_path):
            print(f"Error: Image file '{image_path}' not found.")
            return

        print(f"Reading image from {image_path}...")
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Failed to load the image.")
            return
        print("Image loaded successfully!")

        # Segment the image
        print("Performing semantic segmentation...")
        pil_image, segmentation_mask = segment_image(image_path, model)
        print("Segmentation completed!")

        # Extract ORB features
        print("Extracting ORB features...")
        keypoints, descriptors = extract_orb_features(image)
        if not keypoints:
            print("No keypoints detected. Check your ORB configuration.")
            return
        print(f"Extracted {len(keypoints)} ORB keypoints.")

        # Label the features
        print("Labeling features with semantic segmentation...")
        labeled_features = label_features(keypoints, segmentation_mask)

        # Compute semantic descriptors
        print("Computing semantic descriptors...")
        num_classes = int(segmentation_mask.max()) + 1  # Ensure number of classes is correct
        semantic_descriptors = compute_semantic_descriptors(segmentation_mask, keypoints, num_classes, threshold=0.2)
        print(f"Computed semantic descriptors for {len(semantic_descriptors)} keypoints.")

        # Optional: Print one example
        print("Example semantic descriptor (first keypoint):")
        print(semantic_descriptors[0] if len(semantic_descriptors) > 0 else "No descriptors computed.")


        # Visualize
        print("Visualizing keypoints...")
        overlayed_image = overlay_segmentation(image, segmentation_mask)
        visualize_keypoints(overlayed_image, keypoints)

        # Save results
        output_path = "results/segmented_overlay.png"
        cv2.imwrite(output_path, overlayed_image)
        print(f"Results saved at {output_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
