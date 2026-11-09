"""3D ResNet-18 encoder (MONAI) with a small classification head.

`pretrained="medicalnet"` loads MedicalNet weights (pre-trained on 23 medical
datasets) through MONAI; they exist for a single input channel, so for CT+PET
the first convolution is widened by sharing the pretrained filters between the
channels. The encoder outputs a 512-value vector per cube, which the head maps
to one logit per output.
"""

from __future__ import annotations

import torch
from torch import nn

FEATURES = 512


def _medicalnet_resnet18(pretrained: bool) -> nn.Module:
    from monai.networks.nets import resnet18

    return resnet18(
        pretrained=pretrained,
        spatial_dims=3,
        n_input_channels=1,
        feed_forward=False,  # no built-in classifier: the encoder returns the 512-d feature vector
        shortcut_type="A",
        bias_downsample=True,
    )


class ResNet3DClassifier(nn.Module):
    def __init__(self, in_channels: int = 1, n_outputs: int = 1, pretrained: str = "none", dropout: float = 0.3):
        super().__init__()
        self.encoder = _medicalnet_resnet18(pretrained == "medicalnet")
        if in_channels != 1:
            old = self.encoder.conv1
            new = nn.Conv3d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
            with torch.no_grad():
                new.weight.copy_(old.weight.repeat(1, in_channels, 1, 1, 1) / in_channels)
                if old.bias is not None:
                    new.bias.copy_(old.bias)
            self.encoder.conv1 = new
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(FEATURES, n_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
