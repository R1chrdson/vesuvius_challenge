import numpy as np
from sklearn.model_selection import KFold
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryFBetaScore
from tqdm import tqdm, trange

import wandb
from source.helpers.augmentations import transform
from source.helpers.config import TRAINING_KEYS, Config
from source.helpers.dataset import (ApplyTransformDataset,
                                    VesuviusDummyDataSet,
                                    VesuviusOriginalDataSet)
from source.helpers.early_stopper import EarlyStopping
from source.helpers.logger import logger
from source.helpers.losses import LOSSES
from source.helpers.utils import prepare_folders, seed_everything
from source.models import MODELS


def train_batch(batch, model, optimizer, criterion, metrics):
    x, y = batch
    x = x.to(Config.DEVICE)
    y = y.to(Config.DEVICE)
    optimizer.zero_grad()
    y_hat = model(x)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()
    metrics.update(y_hat, y)
    return loss.item()


def val_batch(batch, model, criterion, metrics):
    x, y = batch
    x = x.to(Config.DEVICE)
    y = y.to(Config.DEVICE)
    y_hat = model(x)
    loss = criterion(y_hat, y)
    metrics.update(y_hat, y)
    return loss.item()


def train_epoch(data_loader, model, optimizer, criterion, metrics):
    model.train()
    losses = []
    with tqdm(total=len(data_loader), desc="Train") as pbar:
        for batch in data_loader:
            loss_value = train_batch(batch, model, optimizer, criterion, metrics)
            losses.append(loss_value)
            pbar.set_postfix(loss=f'{loss_value:5.3f}')
            pbar.update()

        metric_data = {
            "loss": np.mean(losses),
            **{
                metric_name: metric_value.cpu().item()
                for metric_name, metric_value in metrics.compute().items()
            },
        }
        pbar.set_postfix(metric_data)
        metrics.reset()
    return metric_data


def test_epoch(data_loader, model, criterion, metrics):
    model.eval()
    losses = []
    with tqdm(total=len(data_loader), desc="Test") as pbar:
        for batch in data_loader:
            loss_value = val_batch(batch, model, criterion, metrics)
            losses.append(loss_value)
            pbar.set_postfix(loss=f'{loss_value:5.3f}')
            pbar.update()

        metric_data = {
            "loss": np.mean(losses),
            **{
                metric_name: metric_value.item()
                for metric_name, metric_value in metrics.compute().items()
            },
        }
        pbar.set_postfix(metric_data)
        metrics.reset()
    return metric_data


def fit_model(train_loader, test_loader, comment=""):
    model = MODELS[Config.MODEL]["model"]().to(Config.DEVICE)
    criterion = LOSSES[Config.LOSS]()
    optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    early_stopping = EarlyStopping(patience=Config.PATIENCE, verbose=True)

    metrics = MetricCollection([
        BinaryAccuracy(),
        BinaryFBetaScore(beta=0.5),
    ])
    train_metrics = metrics.clone().to(Config.DEVICE)
    test_metrics = metrics.clone().to(Config.DEVICE)

    wandb.init(
        project=Config.WANDB_PROJECT,
        config={key.lower(): Config[key] for key in TRAINING_KEYS},
        tags=[Config.MODEL, comment, Config.ENVIRONMENT],
        name='-'.join(filter(bool, [Config.MODEL, comment, Config.CHECKPOINTS_SLUG])),
        group=Config.CHECKPOINTS_SLUG
    )

    for _ in trange(Config.EPOCHS, desc="Epoch"):
        train_metric_data = train_epoch(train_loader, model, optimizer, criterion, train_metrics)
        test_metric_data = test_epoch(test_loader, model, criterion, test_metrics)

        wandb.log({
            **{f"train/{key}": value for key, value in train_metric_data.items()},
            **{f"test/{key}": value for key, value in test_metric_data.items()},
        })

        early_stopping(test_metric_data["BinaryFBetaScore"], model, comment)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

    wandb.log({"best_val_score": early_stopping.best_score})

    artifact = wandb.Artifact(
        name=early_stopping.best_model_checkpoint_path.stem,
        type="model",
        metadata={
            "train_metrics": train_metric_data,
            "test_metrics": test_metric_data,
        }
    )
    artifact.add_file(early_stopping.best_model_checkpoint_path)
    wandb.log_artifact(artifact)
    wandb.finish()
    return early_stopping.best_score


def train():
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

    fit_model(train_loader, test_loader)


def train_with_cv():
    """
    Train model with cross validation.
    Use `Config.CV_FOLDS` to set number of folds. Default value is 5.
    Use `Config.FOLD_IDX` to set fold index to train. Default value of -1 means train all folds
    """
    logger.info(f"Environment: {Config}")
    logger.info("Starting training with cross validation")
    seed_everything()
    prepare_folders()

    dataset = MODELS[Config.MODEL]["dataset"]()
    train_dataset = ApplyTransformDataset(dataset, transform=transform)

    # Setup augmentations for any dataset except VesuviusOriginalDataSet by default
    if type(dataset) == VesuviusOriginalDataSet:
        train_dataset = dataset

    splits = KFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.SEED)

    fold_data_idxs = list(splits.split(np.arange(len(dataset))))
    fold_idxs  = range(1, Config.CV_FOLDS + 1)

    if Config.FOLD_IDX != -1:
        if isinstance(Config.FOLD_IDX, list):
            fold_data_idxs = [fold_data_idxs[idx - 1] for idx in Config.FOLD_IDX]
            fold_idxs = Config.FOLD_IDX
        elif isinstance(Config.FOLD_IDX, int):
            fold_data_idxs = [fold_data_idxs[Config.FOLD_IDX]]
            fold_idxs = [Config.FOLD_IDX]
        else:
            raise NotImplementedError("Config.FOLD_IDX must be list or int")

    fold_scores = []
    for fold, (train_idx, val_idx) in zip(fold_idxs, fold_data_idxs):
        logger.info(f"Fold {fold}")
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
        )
        test_loader = DataLoader(
            dataset,
            sampler=test_sampler,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
        )

        fold_score = fit_model(train_loader, test_loader, comment=f"fold_{fold}")
        fold_scores.append(fold_score)

    print(f"CV Score: {np.mean(fold_scores)}")


if __name__ == "__main__":
    train_with_cv()
