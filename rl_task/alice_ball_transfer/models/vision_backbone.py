"""ResNet-18 vision backbone for wrist camera observations.

Extracts 512-dim features from stacked RGB frames. Runs as a frozen feature
extractor inside the environment (not part of the RL policy network). ImageNet
pretraining provides general visual features; domain randomization in sim
ensures robustness to real-world appearance.

Usage in env:
    backbone = VisionBackbone(num_frames=4).to(device)
    features = backbone(stacked_images)  # (N, 512)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class VisionBackbone(nn.Module):
    """Frozen ResNet-18 feature extractor for multi-frame RGB input."""

    def __init__(self, num_frames: int = 4, feature_dim: int = 512):
        super().__init__()
        self.num_frames = num_frames
        self.feature_dim = feature_dim

        # Load ImageNet-pretrained ResNet-18
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Modify first conv layer to accept stacked frames (num_frames * 3 channels)
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(
            num_frames * 3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Initialize: average pretrained weights across frame slots
        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.repeat(1, num_frames, 1, 1) / num_frames

        # Build feature extractor (everything before final FC)
        self.features = nn.Sequential(
            new_conv,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,   # Block 1 — 64 channels
            resnet.layer2,   # Block 2 — 128 channels
            resnet.layer3,   # Block 3 — 256 channels
            resnet.layer4,   # Block 4 — 512 channels
            resnet.avgpool,  # Global average pool → (N, 512, 1, 1)
        )

        # Freeze all parameters (feature extractor only)
        for param in self.features.parameters():
            param.requires_grad = False
        self.eval()

        # ImageNet normalization constants
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        )

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract visual features from stacked frames.

        Args:
            images: (N, num_frames*3, H, W) float tensor in [0, 1]

        Returns:
            features: (N, 512) visual feature vector
        """
        # Normalize each frame with ImageNet stats
        # Repeat mean/std for num_frames channels
        mean = self.mean.repeat(1, self.num_frames, 1, 1)
        std = self.std.repeat(1, self.num_frames, 1, 1)
        x = (images - mean) / std

        x = self.features(x)
        return x.flatten(1)  # (N, 512)
