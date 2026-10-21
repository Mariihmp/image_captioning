"""Image feature extractor shared by every model."""

import torch
import torchvision
from torch import nn

IMAGE_SIZE = 256  # gives an 8x8 grid of image regions
FEATURE_DIM = 1280

# EfficientNet was trained on images normalised with these ImageNet values,
# so images must be normalised the same way in training and testing.
IMAGE_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class FeatureExtractor(nn.Module):
    """Frozen EfficientNet-B0. Turns an image into 64 region vectors of size 1280."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        efficientnet = torchvision.models.efficientnet_b0(weights=weights)
        self.model = efficientnet.features
        for param in self.model.parameters():
            param.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, images):
        features = self.model(images)  # (batch, 1280, 8, 8)
        return features.flatten(start_dim=2).permute(0, 2, 1)  # (batch, 64, 1280)
