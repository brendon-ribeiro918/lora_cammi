# from data.preprocess import ImagePreprocessor
# from data.augment import ImageAugmentor
# from model.lora_model import LoRaStableDiffusionModel
# from model.train import Trainer
# from model.evaluate import Evaluator

# def main():
#     input_directory = '/home/emeka/Downloads/Cammi'
#     preprocessed_directory = '/home/emeka/Downloads/Processed'
#     augmented_directory = '/home/emeka/Downloads/Augmented'

#     # Preprocess Images
#     preprocessor = ImagePreprocessor(input_directory, preprocessed_directory)
#     preprocessor.preprocess()

#     # Augment Images
#     augmentor = ImageAugmentor(preprocessed_directory, augmented_directory)
#     augmentor.augment()

#     # Load Dataset
#     # Assuming a DataLoader is implemented to load preprocessed and augmented images
#     data_loader = None

#     # Initialize Model
#     model = LoRaStableDiffusionModel()

#     # Train Model
#     trainer = Trainer(model, data_loader)
#     trainer.train()

#     # Evaluate Model
#     evaluator = Evaluator(model, data_loader)
#     ssim_value, psnr_value = evaluator.evaluate()

#     print(f"SSIM: {ssim_value}, PSNR: {psnr_value}")

# if __name__ == "__main__":
#     main()

from data.preprocess import ImagePreprocessor
from data.augment import ImageAugmentor
from data.data_loader import get_data_loader
from model.lora_model import LoRaStableDiffusionModel
from model.train import Trainer
from model.evaluate import Evaluator

def main():
    input_directory = '/home/emeka/Downloads/Cammi'
    preprocessed_directory = '/home/emeka/Downloads/processed'
    augmented_directory = '/home/emeka/Downloads/augmented'
    checkpoint_directory = '/home/emeka/Downloads/checkpoint'

    # Preprocess Images
    preprocessor = ImagePreprocessor(input_directory, preprocessed_directory)
    preprocessor.preprocess()

    # Augment Images
    augmentor = ImageAugmentor(preprocessed_directory, augmented_directory)
    augmentor.augment()

    # Load Dataset
    train_loader = get_data_loader(augmented_directory)

    # Initialize Model
    model = LoRaStableDiffusionModel()

    # Train Model
    trainer = Trainer(model, train_loader, checkpoint_dir=checkpoint_directory)
    trainer.train()

    # Evaluate Model
    evaluator = Evaluator(model, train_loader)
    ssim_value, psnr_value = evaluator.evaluate()

    print(f"SSIM: {ssim_value}, PSNR: {psnr_value}")

if __name__ == "__main__":
    main()

