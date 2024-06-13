from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(image_dir, '*'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, image

def get_data_loader(image_dir, batch_size=8, img_size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    dataset = CustomDataset(image_dir, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

