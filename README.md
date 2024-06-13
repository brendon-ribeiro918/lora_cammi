# Stable Diffusion LoRA Training

## Objective
Train a Stable Diffusion LoRA model on a provided set of images and evaluate its performance.

## Table of Contents
1. Environment Setup
2. Training Steps
3. Evaluate the Model
4. Challenges

## Environment Setup
1. **Clone the Repository**
    git clone https://github.com/yourusername/lora_cammi.git
    cd lora_cammi

2. **Install Dependencies**
    Ensure you have Python 3.x installed. Install the required Python packages using pip:
    torch>=1.9.0
    torchvision>=0.10.0
    transformers>=4.11.3
    diffusion>=0.4.0
    numpy>=1.21.2
    matplotlib>=3.4.3
    Pillow>=8.3.2

3. **Run the main script**
    ```bash
    python main.py
    ```
Prepare Data:
Place your low-resolution images in a directory and specify the input directory path in the code.

## Training Steps
Initialize Model:
The LoRA model is initialized with specified parameters such as input channels, hidden channels, number of layers, and kernel size.

Prepare Dataset:
The dataset is prepared using the CustomImageDataset class, which loads the images and applies transformations.

Train the Model:
The model is trained using Mean Squared Error (MSE) loss and Adam optimizer. Training progresses over multiple epochs.

Save the Model:
After training, the model state is saved to a file (lora_model.pth) for future use.

## Evaluate the Model
Evaluation Metrics=>

The model's performance is evaluated using two metrics:

    Peak Signal-to-Noise Ratio (PSNR)
    Structural Similarity Index (SSIM)

Evaluation Process:
```The trained model is used to generate high-resolution images from the low-resolution inputs.
PSNR and SSIM are calculated between the generated images and the original high-resolution images.
```
## Challenges
