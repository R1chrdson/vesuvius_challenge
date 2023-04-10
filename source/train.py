from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from tqdm import trange, tqdm

from source.helpers.config import Config
from source.helpers.dataset import VesuviusDummyDataSet
from source.helpers.logger import logger
from source.helpers.utils import seed_everything
from source.models import MODELS
from source.helpers.early_stopper import EarlyStopping
from time import sleep


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
    criterion = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE)
    early_stopping = EarlyStopping(patience=Config.PATIENCE, verbose=True)

    for epoch in trange(Config.EPOCHS, desc="Epoch"):
        with tqdm(len(train_loader), desc="Train") as pbar:
            for batch in train_loader:
                x, y = batch
                x = x.to(Config.DEVICE)
                y = y.to(Config.DEVICE)
                print("x", x.shape, "y", y.shape)
                optimizer.zero_grad()
                y_hat = model(x)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

        with tqdm(len(test_loader), desc="Test") as pbar:
            for batch in test_loader:
                x, y = batch
                x = x.to(Config.DEVICE)
                y = y.to(Config.DEVICE)

                y_hat = model(x)
                loss = criterion(y_hat, y)

                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

        early_stopping(loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
