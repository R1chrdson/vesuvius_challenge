"""Dataset classes used for training and evaluation of the model."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import cv2

from .config import Config
from .logger import logger


def is_slice_to_load(slice_path: Path) -> bool:
    try:
        return Config.Z_START <= int(slice_path.stem) < Config.Z_START + Config.Z_NUMBER
    except ValueError:
        return False


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
            0, 255, (n_samples, Config.Z_NUMBER, Config.TILE_SIZE, Config.TILE_SIZE)
        )

        # 1 if average of all layers per pixel is greater than 128, 0 otherwise
        self.labels = np.mean(self.voxels_data, axis=(1, 2, 3)) > 128

    def __len__(self) -> int:
        return self.voxels_data.shape[0]

    def __getitem__(self, index):
        voxel = (self.voxels_data[index]).astype(np.float32) / 255.
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

        voxels = {}
        labels = {}
        for fragment in tqdm(train_fragment_paths):
            voxels[fragment], labels[fragment] = self._load_fragment(fragment)

        self.voxels_data = np.concatenate([voxels.pop(fragment) for fragment in train_fragment_paths], axis=0)
        self.labels = np.concatenate([labels.pop(fragment) for fragment in train_fragment_paths], axis=0)

    def _load_fragment(self, fragment_path: Path):
        slice_paths = sorted(list((fragment_path / "surface_volume").glob("*.tif")))
        slice_paths = filter(is_slice_to_load, slice_paths)
        labels_path = fragment_path / "inklabels.png"
        labels_img = cv2.imread(str(labels_path), cv2.IMREAD_GRAYSCALE).astype(bool)
        mask = cv2.imread(str(fragment_path / "mask.png"), cv2.IMREAD_GRAYSCALE).astype(bool)
        masked_idxs = self._get_masked_idxs(mask)
        voxels_data = np.empty((len(masked_idxs), Config.Z_NUMBER, Config.TILE_SIZE, Config.TILE_SIZE), dtype=np.uint8)
        for i, slice_path in enumerate(tqdm(slice_paths, leave=False)):
            # In this case, we use cv2 to load image, because it's faster than PIL
            slice_img = cv2.imread(str(slice_path), cv2.IMREAD_UNCHANGED)

            # Convert to uint8 to save memory usage
            slice_data = (slice_img // 255).astype(np.uint8)

            voxels_data[:, i, :, :] = self._split_slice(slice_data, masked_idxs)

        labels_data = self._split_slice(labels_img, masked_idxs)
        labels_data = labels_data.mean(axis=(1, 2)) > 0.5
        return voxels_data, labels_data

    def _get_masked_idxs(self, mask):
        """
        Returns list of tuples with indexes of tiles with data.
        Basically, the idea of this function is to pre calculate the indexes of tiles with data,
        so it would be possible to pre-allocate memory for the fragments and then just fill it with data.
        This approach is much faster and less memory consuming than just appending to the list and concatenating then.
        """
        mask_idxs = []
        for i in range(0, mask.shape[0], Config.TILE_SIZE):
            for j in range(0, mask.shape[1], Config.TILE_SIZE):
                if mask[i : i + Config.TILE_SIZE, j : j + Config.TILE_SIZE].any():
                    mask_idxs.append((i, j))
        return mask_idxs

    def _split_slice(self, slice_data, masked_idxs):
        """Split slice into tiles. It's possible to mask to filter out tiles with no data."""
        tiles = np.empty((len(masked_idxs), Config.TILE_SIZE, Config.TILE_SIZE), dtype=np.uint8)
        for k, (i, j) in enumerate(masked_idxs):
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

            tiles[k] = tile

        return tiles

    def __len__(self) -> int:
        return self.voxels_data.shape[0]

    def __getitem__(self, index):
        voxel = (self.voxels_data[index]).astype(np.float32) / 255.
        label = np.expand_dims(self.labels[index], axis=0)
        return torch.from_numpy(voxel).unsqueeze(0), torch.FloatTensor(label)


class UnetVesuviusDataset(VesuviusOriginalDataSet):
    def _load_fragment(self, fragment_path: Path):
        slice_paths = sorted(list((fragment_path / "surface_volume").glob("*.tif")))
        slice_paths = filter(is_slice_to_load, slice_paths)
        labels_path = fragment_path / "inklabels.png"
        labels_img = cv2.imread(str(labels_path), cv2.IMREAD_GRAYSCALE).astype(bool)
        mask = cv2.imread(str(fragment_path / "mask.png"), cv2.IMREAD_GRAYSCALE).astype(bool)
        masked_idxs = self._get_masked_idxs(mask)
        voxels_data = np.empty((len(masked_idxs), Config.TILE_SIZE, Config.TILE_SIZE, Config.Z_NUMBER), dtype=np.uint8)
        for i, slice_path in enumerate(tqdm(slice_paths, leave=False)):
            # In this case, we use cv2 to load image, because it's faster than PIL
            slice_img = cv2.imread(str(slice_path), cv2.IMREAD_UNCHANGED)

            # Convert to uint8 to save memory usage
            slice_data = (slice_img // 255).astype(np.uint8)

            voxels_data[..., i] = self._split_slice(slice_data, masked_idxs)

        mu = np.mean(voxels_data, axis=(0, 1, 2))
        std = np.std(voxels_data, axis=(0, 1, 2))

        voxels_data = (voxels_data - mu) / std
        voxels_data = np.transpose(voxels_data, (0, 3, 1, 2))

        labels_data = self._split_slice(labels_img, masked_idxs)
        return voxels_data, labels_data

    def __getitem__(self, index):
        voxel = (self.voxels_data[index]).astype(np.float32) / 255.
        label = np.expand_dims(self.labels[index], axis=0)
        return torch.from_numpy(voxel), torch.FloatTensor(label)


class ApplyTransformDataset(Dataset):
    """This class is used to apply transforms to the dataset.
    Only classes with explicit masks are supported, so VesuviusOriginalDataSet is not supported!
    """
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        voxel = self.dataset.voxels_data[index]
        label = self.dataset.labels[index]
        transformed = self.transform(image=voxel.transpose(1, 2, 0), mask=label)
        transformed_voxel = transformed['image'].transpose(2, 0, 1).astype(np.float32) / 255.
        transformed_label = np.expand_dims(transformed['mask'], axis=0)
        return torch.from_numpy(transformed_voxel), torch.FloatTensor(transformed_label)
