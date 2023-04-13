import logging
from tqdm import tqdm

from .config import Config


logger = logging.getLogger(__name__)
logger.setLevel(Config.LOG_LEVEL)
stream_handler = logging.StreamHandler()
stream_handler.setStream(tqdm)
logger.addHandler(stream_handler)
