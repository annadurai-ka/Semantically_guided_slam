import cv2
import numpy as np

class Frame:
    def __init__(self, image, sem_image, extractor, K=None, dist_coeff=None):
        """
        Represents a single image frame for SLAM processing.
        Args:
            image (np.ndarray): Grayscale input image.
            sem_image (np.ndarray): Semantic label image (same size as image).
            extractor (cv2.Feature2D): Feature extractor (e.g., ORB, SIFT).
            K (np.ndarray): 3x3 intrinsic matrix (optional).
            dist_coeff (np.ndarray): Distortion coefficients (optional).
        """
        self.image = image
        self.sem_image = sem_image
        self.K = K
        self.dist_coeff = dist_coeff

        # Detect keypoints and descriptors
        self.keypoints, self.descriptors = extractor.detectAndCompute(self.image, None)
        if self.keypoints is None:
            self.keypoints = []
            self.descriptors = np.array([], dtype=np.uint8).reshape(0, 32)

        # Undistort keypoint coordinates if needed
        if self.K is not None and self.dist_coeff is not None and len(self.keypoints) > 0:
            pts = np.array([kp.pt for kp in self.keypoints], dtype=np.float32).reshape(-1, 1, 2)
            pts_undist = cv2.undistortPoints(pts, self.K, self.dist_coeff, P=self.K)
            for i, pt_u in enumerate(pts_undist):
                self.keypoints[i].pt = (float(pt_u[0][0]), float(pt_u[0][1]))

        # Compute semantic descriptors
        self.semantic_descriptors = self.compute_semantic_descriptors()

    def compute_semantic_descriptors(self, patch_radius=15):
        """
        Builds semantic descriptors for each keypoint as a histogram of label IDs in a local patch.
        Args:
            patch_radius (int): Radius of square patch (default 15 → 31x31).
        Returns:
            List of 1D numpy arrays (histograms).
        """
        h, w = self.sem_image.shape[:2]
        sem_descs = []
        for kp in self.keypoints:
            x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
            x0, x1 = max(0, x - patch_radius), min(w, x + patch_radius + 1)
            y0, y1 = max(0, y - patch_radius), min(h, y + patch_radius + 1)
            patch = self.sem_image[y0:y1, x0:x1]

            if patch.size == 0:
                sem_descs.append(np.zeros(1, dtype=np.float32))
                continue

            classes, counts = np.unique(patch, return_counts=True)
            max_class = int(patch.max())
            hist = np.zeros(max_class + 1, dtype=np.float32)
            hist[classes] = counts
            sem_descs.append(hist)
        return sem_descs
