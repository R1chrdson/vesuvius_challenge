"""Dataset classes used for training and evaluation of the model."""
import numpy as np
import torch
from torch.utils.data import Dataset
from .config import Config


class VesuviusDummyDataSet(Dataset):
    """Vesuvius Dummy Dataset
    This dataset class represents the data format used in the Vesuvius Challenge,
    also there is layout of Dataset class and main idea of Dataset used for training.

    Args:
        n_samples (int, optional): Number of samples to generate. Defaults to 1000.
            This is just for example and should'nt be used in other classes.
    """

    def __init__(self, n_samples: int = 1000):
        # The reason, why I placed main data loading logic here is that
        # Often, the data initially loaded in the form of a file, and then in getitem only returned by the index
        # But it's not always the case, so you can load the data in the getitem method from the file too
        self.volume_data = np.random.randint(
            0, 255, (n_samples, 65, Config.TILE_SIZE, Config.TILE_SIZE)
        )

        # 1 if average of all layers per pixel is greater than 128, 0 otherwise
        self.mask = np.mean(self.volume_data, axis=(1)) > 128

    def __len__(self) -> int:
        return self.volume_data.shape[0]

    def __getitem__(self, index):
        subvolume = self.volume_data[index] / 255.
        label = self.mask[index]
        return torch.from_numpy(subvolume).unsqueeze(0), torch.FloatTensor([label])
