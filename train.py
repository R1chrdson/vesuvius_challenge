from helpers.config import Config
from helpers.dataloader import DataLoader
from helpers.logger import logger
from helpers.utils import seed_everything

if __name__ == "__main__":
    logger.info(f"Environment: {Config}")
    logger.info("Starting training")
    seed_everything(Config.SEED)
