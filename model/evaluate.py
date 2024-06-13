import numpy as np
from skimage.metrics import structural_similarity as ssim

class Evaluator:
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

    def evaluate(self):
        ssim_values = []
        psnr_values = []
        for original_image, generated_image in self.data_loader:
            original = np.array(original_image)
            generated = np.array(generated_image)

            # Evaluate Structural Similarity Index Measure
            ssim_value = ssim(original, generated, multichannel=True)
            ssim_values.append(ssim_value)

            # Evaluate Peak Signal-to-Noise Ratio
            psnr_value = self.psnr(original, generated)
            psnr_values.append(psnr_value)

        return np.mean(ssim_values), np.mean(psnr_values)

    def psnr(self, img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))