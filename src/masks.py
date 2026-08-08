# import cv2
# import numpy as np
# import random

# class MaskGenerator:
#     def __init__(self, image_size=(256, 256)):
#         self.h, self.w = image_size

#     def generate_semantic_mask(self, region_pts):
#         """
#         Generates a 256x256 binary mask (255 = masked region, 0 = visible region)
#         from a set of region landmark points using a convex hull.
#         """
#         mask = np.zeros((self.h, self.w), dtype=np.uint8)
#         if region_pts is not None and len(region_pts) > 0:
#             pts_int = region_pts.astype(np.int32)
#             hull = cv2.convexHull(pts_int)
#             cv2.fillPoly(mask, [hull], 255)
#         return mask

#     def generate_random_box_mask(self, coverage=0.25):
#         """
#         Generates a 256x256 random rectangular mask covering ~coverage fraction of the image.
#         """
#         mask = np.zeros((self.h, self.w), dtype=np.uint8)
#         target_area = self.h * self.w * coverage
#         aspect_ratio = random.uniform(0.5, 2.0)
        
#         box_h = int(np.sqrt(target_area / aspect_ratio))
#         box_w = int(box_h * aspect_ratio)

#         box_h = min(box_h, self.h)
#         box_w = min(box_w, self.w)

#         y1 = random.randint(0, max(0, self.h - box_h))
#         x1 = random.randint(0, max(0, self.w - box_w))

#         mask[y1:y1 + box_h, x1:x1 + box_w] = 255
#         return mask

import cv2
import numpy as np
import random


class MaskGenerator:
    """Mask Generator used for dataset preprocessing and dynamic inference."""

    def __init__(self, image_size=(256, 256)):
        self.h, self.w = image_size

    def generate_semantic_mask(self, region_pts):
        """Generates a binary mask from landmark region points using convex hull."""
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        if region_pts is not None and len(region_pts) > 0:
            pts_int = region_pts.astype(np.int32)
            hull = cv2.convexHull(pts_int)
            cv2.fillPoly(mask, [hull], 255)
        return mask

    def generate_random_box_mask(self, coverage=0.25):
        """Generates a random rectangular mask covering ~coverage fraction of image."""
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        target_area = self.h * self.w * coverage
        aspect_ratio = random.uniform(0.5, 2.0)

        box_h = int(np.sqrt(target_area / aspect_ratio))
        box_w = int(box_h * aspect_ratio)

        box_h = min(box_h, self.h)
        box_w = min(box_w, self.w)

        y1 = random.randint(0, max(0, self.h - box_h))
        x1 = random.randint(0, max(0, self.w - box_w))

        mask[y1 : y1 + box_h, x1 : x1 + box_w] = 255
        return mask