import logging

from helpers.config import Config


logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)
logger.setLevel(Config.LOG_LEVEL)
