from torch.utils.data import DataLoader

from source.helpers.config import Config
from source.helpers.dataset import VesuviusDummyDataSet
from source.helpers.logger import logger
from source.helpers.utils import seed_everything
from source.models import MODELS


if __name__ == "__main__":
    logger.info(f"Environment: {Config}")
    logger.info("Starting training")
    seed_everything(Config.SEED)

    train_dataset = VesuviusDummyDataSet(n_samples=800)
    test_dataset = VesuviusDummyDataSet(n_samples=100)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        shuffle=False,
    )

    model = MODELS[Config.MODEL]().to(Config.DEVICE)
    print(model)
