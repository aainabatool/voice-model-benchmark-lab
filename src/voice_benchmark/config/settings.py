"""Application configuration, loaded from environment variables / .env.

See .env.example for the full list of supported variables.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "INFO"

    # Defaults to local SQLite so the project runs with zero external setup.
    # Point this at Postgres (see .env.example) for anything beyond local dev.
    database_url: str = "sqlite:///./voice_benchmark.db"

    artifacts_dir: Path = Path("./artifacts")
    datasets_dir: Path = Path("./datasets")

    default_device: str = "auto"

    @property
    def is_development(self) -> bool:
        return self.app_env.lower() == "development"


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor -- import and call this, don't instantiate Settings() directly."""
    return Settings()
