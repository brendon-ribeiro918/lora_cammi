# Stable Diffusion LoRA Training

## Objective
Train a Stable Diffusion LoRA model on a provided set of images and evaluate its performance.

## Table of Contents
1. Environment Setup
2. Directory Structure
3. Training Steps
4. Evaluate the Model
5. Challenges

## Environment Setup
1. **Set Up Virtual Environment**
    ```bash
    python -m venv lora_env
    source lora_env/bin/activate  # On Windows use `lora_env\Scripts\activate`
    ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the main script**
    ```bash
    python main.py
    ```

## Directory Structure
    stable_diffusion_lora_training/
    │
    ├── README.md
    ├── requirements.txt
    ├── main.py
    │
    ├── data/
    │   ├── preprocess.py
    │   ├── augment.py
    │   └── data_loader.py
    │
    ├── model/
    │   ├── lora_model.py
    │   ├── train.py
    │   └── evaluate.py

## Training Steps
1. **Define Model and Set Hyperparameters**
    ```python
    class Trainer:
    def __init__(self, model, train_loader, val_loader=None, num_epochs=10, learning_rate=1e-4, checkpoint_dir='checkpoints'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.checkpoint_dir = checkpoint_dir

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    ```
2. **Implement Training Loop**
    ```python
    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for inputs, targets in tqdm(self.train_loader):
                inputs, targets = inputs.to("cuda"), targets.to("cuda")
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}")

            self.save_checkpoint(epoch, avg_loss)
            self.log_metrics(epoch, avg_loss)
    ```

## Evaluate the Model
1. **Generate Images**
    Generate images using the trained model, compare generated images to original images, and provide evaluation metrics (PSNR, SSIM) in the evaluate.py script.
2. **Compare and Evaluate**
    ```python
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
    ```

## Challenges
1. **Hardware Limitations**
    I was not able to test the model properly because my hardware didn’t meet the requirements. This made it challenging to conduct extensive training and evaluation locally.
2. **RunPod Setup Issues**
    Although a RunPod instance was provided to me, I could not set it up correctly as I kept running into errors and had to put it aside in order to meet the deadline. This is because I am not familiar with RunPod, and it is something I will have to learn now.
3. **Evaluation Constraints**
    Due to hardware limitations, I was unable to perform extensive evaluations using metrics like PSNR and SSIM. Evaluations were limited to small sample sizes.
4. **Large Dependencies**
    Downloading and installing large dependencies such as PyTorch and pre-trained models was problematic due to limited bandwidth and storage space. This hindered the setup process.