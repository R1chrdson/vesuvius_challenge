from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from tqdm import trange, tqdm
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryFBetaScore

from source.helpers.config import Config
from source.helpers.dataset import VesuviusDummyDataSet
from source.helpers.logger import logger
from source.helpers.utils import seed_everything, prepare_folders
from source.models import MODELS
from source.helpers.early_stopper import EarlyStopping



if __name__ == "__main__":
    logger.info(f"Environment: {Config}")
    logger.info("Starting training")
    seed_everything()
    prepare_folders()

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

    metrics = MetricCollection([BinaryFBetaScore(beta=0.5)]).to(Config.DEVICE)
    train_metrics = metrics.clone()
    test_metrics = metrics.clone()

    for epoch in trange(Config.EPOCHS, desc="Epoch"):
        with tqdm(total=len(train_loader), desc="Train") as pbar:
            for batch in train_loader:
                x, y = batch
                x = x.to(Config.DEVICE)
                y = y.to(Config.DEVICE)
                optimizer.zero_grad()
                y_hat = model(x)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                train_metrics.update(y_hat, y)

                loss_value = loss.item()
                pbar.set_postfix(loss=loss_value)
                pbar.update()

            pbar.set_postfix(
                loss=loss_value,
                **{
                    metric_name: metric_value.cpu().item()
                    for metric_name, metric_value in train_metrics.compute().items()
                },
            )

        with tqdm(total=len(test_loader), desc="Test") as pbar:
            for batch in test_loader:
                x, y = batch
                x = x.to(Config.DEVICE)
                y = y.to(Config.DEVICE)

                y_hat = model(x)
                loss = criterion(y_hat, y)
                test_metrics.update(y_hat, y)
                loss_value = loss.item()
                pbar.set_postfix(loss=loss_value)
                pbar.update()
            pbar.set_postfix(
                loss=loss_value,
                **{
                    metric_name: metric_value.cpu().item()
                    for metric_name, metric_value in test_metrics.compute().items()
                },
            )

        early_stopping(loss, model, epoch)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
