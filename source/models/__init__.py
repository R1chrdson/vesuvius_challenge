from source.models.example_model import InkDetector
from source.models.unet import UNet
from source.models.pvt import PVTTiny, PVTSmall, PVTMedium, PVTLarge

from source.helpers.dataset import VesuviusOriginalDataSet, UnetVesuviusDataset

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
}
