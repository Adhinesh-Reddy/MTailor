import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

class PreprocessWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

    def forward(self, img_tensor):
        # Assume input is a uint8 tensor image of shape [B, H, W, C]
        x = img_tensor.permute(0, 3, 1, 2).float() / 255.0  # NHWC -> NCHW
        x = T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])(x)
        return self.model(x)
