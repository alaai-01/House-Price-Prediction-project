"""Application settings, loaded from environment / .env (pydantic-settings)."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/  (two levels up from this file: app/core/config.py -> app -> backend)
BACKEND_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BACKEND_DIR / "models"


class Settings(BaseSettings):
    # protected_namespaces=() lets us use `model_*` field names without pydantic warnings.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    app_name: str = "House Price Prediction API"
    app_version: str = "1.0.0"

    # Comma-separated CORS origins. Vite dev server defaults to 5173.
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # Artifacts produced by the notebook (see notebooks/house_price_model.ipynb).
    model_path: Path = MODELS_DIR / "house_price.pkl"
    schema_path: Path = MODELS_DIR / "model_columns.json"
    locations_path: Path = MODELS_DIR / "locations.json"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached so the .env file is parsed only once."""
    return Settings()
