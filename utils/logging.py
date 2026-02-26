"""Structured logging configuration."""

import logging
from logging.config import dictConfig

from config.settings import get_settings


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
        }
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}


_configured = False


def configure_logging() -> None:
    """Apply logging configuration once."""

    global _configured
    if _configured:
        return

    settings = get_settings()
    LOGGING_CONFIG["root"]["level"] = settings.get("LOG_LEVEL", "INFO")
    dictConfig(LOGGING_CONFIG)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with ensured configuration."""

    configure_logging()
    return logging.getLogger(name)


__all__ = ["get_logger", "configure_logging"]

