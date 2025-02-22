import logging

from .logger import logger


def test_logger(caplog):
    with caplog.at_level(logging.INFO):
        log_message = "This is log test"
        logger.info(log_message)
        assert log_message in caplog.text
