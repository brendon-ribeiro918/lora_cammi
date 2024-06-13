import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm


class LoRAModel(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=64, num_layers=5, kernel_size=3):
        super(LoRAModel, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=kernel_size//2))

        for _ in range(num_layers - 2):
            self.layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2))

        self.layers.append(nn.Conv2d(hidden_channels, input_channels, kernel_size, padding=kernel_size//2))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image


def preprocess_images(input_dir, output_dir, image_size=256):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_list = os.listdir(input_dir)

    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    for image_name in image_list:
        image_path = os.path.join(input_dir, image_name)
        img = Image.open(image_path)
        img = preprocess(img)
        img = transforms.ToPILImage()(img)
        save_path = os.path.join(output_dir, image_name)
        img.save(save_path)


def evaluate_model(model, dataloader, device):
    psnr_values = []
    ssim_values = []
    for images in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
        outputs = outputs.clamp(0, 1)
        for i in range(len(images)):
            original_img = images[i].permute(1, 2, 0).cpu().numpy()
            generated_img = outputs[i].permute(1, 2, 0).cpu().numpy()
            psnr = PSNR(original_img, generated_img)
            ssim = SSIM(original_img, generated_img, multichannel=True)
            psnr_values.append(psnr)
            ssim_values.append(ssim)
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    return avg_psnr, avg_ssim


batch_size = 16
num_epochs = 20
learning_rate = 0.001
image_size = 256
input_channels = 3
hidden_channels = 64
num_layers = 5
kernel_size = 3

## Preprocess images ##
input_dir = "/content/sample_data/images.zip"
output_dir = "preprocessed_images"
preprocess_images(input_dir, output_dir, image_size=image_size)

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = CustomImageDataset(root_dir=output_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LoRAModel(input_channels=input_channels, hidden_channels=hidden_channels, num_layers=num_layers, kernel_size=kernel_size)
model.to(device)

criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = batch.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

torch.save(model.state_dict(), "lora_model.pth")

# Generate new images using the trained LoRA model
generated_images_dir = "generated_images"
if not os.path.exists(generated_images_dir):
    os.makedirs(generated_images_dir)

eval_dataset = CustomImageDataset(root_dir=output_dir, transform=transform)
eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

for idx, image in enumerate(eval_dataloader):
    image = image.to(device)
    with torch.no_grad():
        generated_image = model(image)
    generated_image = generated_image.clamp(0, 1)
    save_image(generated_image, os.path.join(generated_images_dir, f"generated_image_{idx}.png"))

eval_dataset_generated = CustomImageDataset(root_dir=generated_images_dir, transform=transform)
eval_dataloader_generated = DataLoader(eval_dataset_generated, batch_size=1, shuffle=False)

avg_psnr, avg_ssim = evaluate_model(model, eval_dataloader_generated, device)
print(f"Average PSNR: {avg_psnr}, Average SSIM: {avg_ssim}")
