import logging

logger = logging.getLogger(__name__)
logger.setLevel("INFO")
formatter = logging.Formatter(
    "{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler = logging.StreamHandler()
console_handler.setLevel("INFO")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
