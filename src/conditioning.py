import cv2
import numpy as np

class ConditioningGenerator:
    def __init__(self, image_size=(256, 256), sigma=2.0):
        self.h, self.w = image_size
        self.sigma = sigma

    def generate_heatmap(self, landmarks, mask=None):
        """
        Renders a 256x256 1-channel landmark conditioning map.
        Only keeps landmarks that fall OUTSIDE the masked region (mask == 0).
        """
        heatmap = np.zeros((self.h, self.w), dtype=np.float32)
        if landmarks is None or len(landmarks) == 0:
            return (heatmap * 255).astype(np.uint8)

        for (x, y) in landmarks:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < self.w and 0 <= iy < self.h:
                # If a mask is provided, ignore landmarks inside the masked area
                if mask is not None and mask[iy, ix] > 0:
                    continue
                heatmap[iy, ix] = 1.0

        # Apply Gaussian Blur to turn point coordinates into continuous structural spatial priors
        if self.sigma > 0:
            heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)
            max_val = heatmap.max()
            if max_val > 0:
                heatmap = heatmap / max_val

        return (heatmap * 255).astype(np.uint8)