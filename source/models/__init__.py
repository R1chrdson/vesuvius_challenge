from source.models.example_model import InkDetector
from source.models.unet import UNet
from source.models.pvt import PVTTiny, PVTSmall, PVTMedium, PVTLarge
from source.models.resnet import ResNet34Unet, ResNet50Unet
from source.models.effnet import EffNet, EffNetBXUnetImagenetPretrained, EffNetBXUnetNoPretrain

from source.helpers.dataset import VesuviusOriginalDataSet, UnetVesuviusDataset, EffNetVesuviusDataset

MODELS = {
    "InkDetector": {
        "model": InkDetector,
        "dataset": VesuviusOriginalDataSet,
    },
    "UNet": {
        "model": UNet,
        "dataset": UnetVesuviusDataset,
    },
    "PVTTiny": {
        "model": PVTTiny,
        "dataset": UnetVesuviusDataset,
    },
    "PVTSmall": {
        "model": PVTSmall,
        "dataset": UnetVesuviusDataset,
    },
    "PVTMedium": {
        "model": PVTMedium,
        "dataset": UnetVesuviusDataset,
    },
    "PVTLarge": {
        "model": PVTLarge,
        "dataset": UnetVesuviusDataset,
    },
    "ResNet34Unet": {
        "model": ResNet34Unet,
        "dataset": UnetVesuviusDataset,
    },
    "ResNet50Unet": {
        "model": ResNet50Unet,
        "dataset": UnetVesuviusDataset,
    },
    "EffNet": {
        "model": EffNet,
        "dataset": EffNetVesuviusDataset,
    },
    "EffNetBXUnetImagenetPretrained": {
        "model": EffNetBXUnetImagenetPretrained,
        "dataset": UnetVesuviusDataset,
    },
    "EffNetBXUnetNoPretrain": {
        "model": EffNetBXUnetNoPretrain,
        "dataset": UnetVesuviusDataset,
    }
}
