import torchvision.models as models
import torch.nn as nn

from source.helpers.config import Config


# ================= EfficientNet ========================

class EffNet:
    def __new__(cls):
        fine_tune = 1 == Config.EFFNET_FINE_TUNE
        version = Config.EFFNET_VERSION

        model = build_efficient_net_model(version, fine_tune)

        in_features = {
            'b1': 1280,
            'b2': 1408,
            'b3': 1536,
            'b4': 1792,
            'b5': 2048,
            'b6': 2304,
            'b7': 2560
        }

        in_features_first = in_features.get(version)
        out_features_first = int(in_features.get(version) / 2)

        model.classifier = nn.Sequential(
            nn.Linear(in_features=in_features_first, out_features=out_features_first),
            nn.ReLU(),  # ReLu to be the activation function
            nn.Dropout(p=Config.DROPOUT),
            nn.Linear(in_features=out_features_first, out_features=320),
            nn.ReLU(),
            nn.Linear(in_features=320, out_features=2),
        )

        return model


def build_efficient_net_model(model_version, fine_tune=True, print_info=True):
    model = getattr(models, 'efficientnet_' + model_version)()

    if fine_tune:
        print_info and print('[INFO]: Fine-tuning all layers...')

        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print_info and print('[INFO]: Freezing hidden layers...')

        for params in model.parameters():
            params.requires_grad = False

    return model
