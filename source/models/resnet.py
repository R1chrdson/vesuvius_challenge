from typing import List, Optional, Union
import segmentation_models_pytorch as smp
import torch.nn as nn

from source.helpers.config import Config


class ResNet34Unet(smp.Unet):
    def __init__(self):
        super().__init__(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=Config.Z_NUMBER,
        classes=1,
        activation=nn.Sigmoid,
    )


class ResNet50Unet(smp.Unet):
    def __init__(self):
        super().__init__(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=Config.Z_NUMBER,
        classes=1,
        activation=nn.Sigmoid,
    )
