import torch.nn as nn
import torch
from diffusers import StableDiffusionPipeline

class LoRaLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super(LoRaLayer, self).__init__()
        self.low_rank_matrix1 = nn.Parameter(torch.randn(in_features, rank))
        self.low_rank_matrix2 = nn.Parameter(torch.randn(rank, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.low_rank_matrix1 @ self.low_rank_matrix2 + self.bias

class LoRaStableDiffusionModel(nn.Module):
    def __init__(self, model_name='CompVis/stable-diffusion-v1-4', lora_rank=4):
        super(LoRaStableDiffusionModel, self).__init__()
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_name).to("cuda")
        
        self.lora_layer1 = LoRaLayer(512, 512, rank=lora_rank)
        self.lora_layer2 = LoRaLayer(512, 512, rank=lora_rank)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.lora_layer1(x)
        x = self.relu(x)
        x = self.lora_layer2(x)
        return self.pipeline(x)

