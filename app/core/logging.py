import logging
import sys

from app.core.config import get_settings


_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging() -> None:
    settings = get_settings()
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(settings.log_level.upper())
        return

    logging.basicConfig(
        level=settings.log_level.upper(),
        format=_LOG_FORMAT,
        stream=sys.stdout,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
