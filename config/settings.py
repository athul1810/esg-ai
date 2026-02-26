"""Application settings loader with environment fallbacks."""

import os
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "storage" / "esg_ai.db"


def _ensure_storage_dir() -> None:
    storage_dir = BASE_DIR / "storage"
    storage_dir.mkdir(exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> dict:
    """Return application configuration values with sensible defaults."""

    _ensure_storage_dir()

    settings = {
        "ENVIRONMENT": os.getenv("ESG_AI_ENV", "development"),
        "DATABASE_URL": os.getenv("ESG_AI_DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}"),
        "LOG_LEVEL": os.getenv("ESG_AI_LOG_LEVEL", "INFO"),
    }

    return settings


__all__ = ["get_settings", "BASE_DIR"]

