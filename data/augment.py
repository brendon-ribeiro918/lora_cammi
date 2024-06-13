import os
from glob import glob
from torchvision import transforms
from PIL import Image

class ImageAugmentor:
    def __init__(self, input_dir, output_dir, img_size=(256, 256)):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.img_size = img_size

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

    def augment(self):
        image_paths = glob(os.path.join(self.input_dir, '*'))
        for img_path in image_paths:
            img = Image.open(img_path)
            img_augmented = self.transform(img)
            output_path = os.path.join(self.output_dir, os.path.basename(img_path))
            img_augmented = transforms.ToPILImage()(img_augmented)
            img_augmented.save(output_path)
