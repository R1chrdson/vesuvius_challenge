"""Dataset classes used for training and evaluation of the model."""
import warnings
from pathlib import Path
import PIL.Image as Image
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import cv2

from .config import Config
from .logger import logger

warnings.simplefilter("ignore", Image.DecompressionBombWarning)


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
        self.voxels_data = np.random.randint(
            0, 255, (n_samples, 65, Config.TILE_SIZE, Config.TILE_SIZE)
        )

        # 1 if average of all layers per pixel is greater than 128, 0 otherwise
        self.labels = np.mean(self.voxels_data, axis=(1, 2, 3)) > 128

    def __len__(self) -> int:
        return self.voxels_data.shape[0]

    def __getitem__(self, index):
        voxel = (self.voxels_data[index] / 255.0).astype(np.float32)
        label = np.expand_dims(self.labels[index], axis=0)
        return torch.from_numpy(voxel).unsqueeze(0), torch.FloatTensor(label)


class VesuviusOriginalDataSet(Dataset):
    """This dataset uses original data from Vesuvius Challenge.
    No optimizations, processing, and separations applied.
    """

    def __init__(self):
        self.voxels_data = None
        self.labels = None
        self._load_data()

    def _load_data(self):
        logger.info("Loading the data...")

        train_data_path = Config.DATASET_PATH / "train"
        train_fragment_paths = sorted(list(train_data_path.iterdir()))

        voxels = []
        labels = []
        for fragment in tqdm(train_fragment_paths):
            voxels_data, labels_data = self._load_fragment(fragment)
            voxels.append(voxels_data)
            labels.append(labels_data)
            # break # TODO REMOVE IT TO GET FULL DATA

        self.voxels_data = np.concatenate(voxels, axis=0)
        self.labels = np.concatenate(labels, axis=0)

    def _load_fragment(self, fragment_path: Path):
        slice_paths = sorted(list((fragment_path / "surface_volume").glob("*.tif")))
        labels_path = fragment_path / "inklabels.png"
        labels_img = np.array(Image.open(labels_path), dtype=bool)
        mask = np.array(Image.open(fragment_path / "mask.png"), dtype=bool)
        voxels = []
        for slice_path in tqdm(slice_paths, leave=False):
            # In this case, we use cv2 to load image, because it's faster than PIL
            slice_img = cv2.imread(str(slice_path), cv2.IMREAD_UNCHANGED)

            # Convert to uint8 to save memory usage
            slice_data = (slice_img // 255).astype(np.uint8)

            tiles = self.split_slice(slice_data, mask)
            voxels.append(tiles)

        voxels_data = np.stack(voxels, axis=1)

        labels_data = self.split_slice(labels_img, mask)
        labels_data = labels_data.mean(axis=(1, 2)) > 0.5
        return voxels_data, labels_data

    def split_slice(self, slice_data, mask=None):
        """Split slice into tiles. It's possible to mask to filter out tiles with no data."""
        tiles = []
        for i in range(0, slice_data.shape[0], Config.TILE_SIZE):
            for j in range(0, slice_data.shape[1], Config.TILE_SIZE):
                if mask is not None:
                    if not mask[
                        i : i + Config.TILE_SIZE, j : j + Config.TILE_SIZE
                    ].any():
                        continue

                tile = slice_data[i : i + Config.TILE_SIZE, j : j + Config.TILE_SIZE]
                if tile.shape != (Config.TILE_SIZE, Config.TILE_SIZE):
                    tile = np.pad(
                        tile,
                        (
                            (0, Config.TILE_SIZE - tile.shape[0]),
                            (0, Config.TILE_SIZE - tile.shape[1]),
                        ),
                        "constant",
                        constant_values=0,
                    )
                tiles.append(tile)

        return np.stack(tiles)

    def __len__(self) -> int:
        return self.voxels_data.shape[0]

    def __getitem__(self, index):
        voxel = (self.voxels_data[index] / 255.0).astype(np.float32)
        label = np.expand_dims(self.labels[index], axis=0)
        return torch.from_numpy(voxel).unsqueeze(0), torch.FloatTensor(label)
