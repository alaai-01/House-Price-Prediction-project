"""Model loading and prediction.

The model is loaded once at startup (see app.main lifespan) and held in a small
singleton so requests don't re-read the 1.8 MB pickle every time.
"""
import json
import logging

import joblib

from app.core.config import get_settings
from app.schemas.prediction import PredictionRequest
from app.services.preprocessing import build_feature_frame

logger = logging.getLogger(__name__)


def format_inr(amount: float) -> str:
    """Format rupees the Indian way: '₹ 1.25 Cr', '₹ 42.50 Lac', '₹ 85,000'."""
    if amount >= 1e7:
        return f"₹ {amount / 1e7:.2f} Cr"
    if amount >= 1e5:
        return f"₹ {amount / 1e5:.2f} Lac"
    return f"₹ {amount:,.0f}"


class ModelService:
    """Holds the fitted Pipeline and the metadata needed for inference."""

    def __init__(self) -> None:
        self._model = None
        self._known_locations: set[str] = set()
        self._sklearn_version: str = "unknown"

    def load(self) -> None:
        settings = get_settings()

        if not settings.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {settings.model_path}. "
                "Run the notebook (notebooks/house_price_model.ipynb) to produce it, "
                "then copy house_price.pkl into backend/models/."
            )

        logger.info("Loading model from %s", settings.model_path)
        self._model = joblib.load(settings.model_path)

        # Known locations come from model_columns.json (the 'location_grouped' options).
        if settings.schema_path.exists():
            schema = json.loads(settings.schema_path.read_text(encoding="utf-8"))
            self._sklearn_version = schema.get("sklearn_version", "unknown")
            opts = schema.get("categorical_options", {}).get("location_grouped", [])
            self._known_locations = {o.lower() for o in opts}
        elif settings.locations_path.exists():
            locs = json.loads(settings.locations_path.read_text(encoding="utf-8"))
            self._known_locations = {o.lower() for o in locs}

        logger.info(
            "Model loaded (sklearn %s, %d known locations).",
            self._sklearn_version,
            len(self._known_locations),
        )

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def sklearn_version(self) -> str:
        return self._sklearn_version

    def predict(self, req: PredictionRequest) -> float:
        if self._model is None:
            raise RuntimeError("Model is not loaded.")
        X = build_feature_frame(req, self._known_locations)
        # The Pipeline is wrapped in a TransformedTargetRegressor(log1p/expm1),
        # so predict() already returns the price in rupees — no manual expm1 needed.
        price = float(self._model.predict(X)[0])
        return max(price, 0.0)


# Module-level singleton, populated at startup.
model_service = ModelService()
