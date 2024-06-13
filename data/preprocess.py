import os
import cv2
import numpy as np
from glob import glob

class ImagePreprocessor:
    def __init__(self, input_dir, output_dir, img_size=(256, 256)):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.img_size = img_size

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def preprocess(self):
        image_paths = glob(os.path.join(self.input_dir, '*'))
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_resized = cv2.resize(img, self.img_size)
            img_normalized = img_resized / 255.0  # Normalize to [0, 1]
            output_path = os.path.join(self.output_dir, os.path.basename(img_path))
            cv2.imwrite(output_path, (img_normalized * 255).astype(np.uint8))
