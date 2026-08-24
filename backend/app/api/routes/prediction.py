"""API routes: GET /health and POST /predict."""
import logging

from fastapi import APIRouter, HTTPException

from app.schemas.prediction import HealthResponse, PredictionRequest, PredictionResponse
from app.services.inference import format_inr, model_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model_service.is_ready else "degraded",
        model_loaded=model_service.is_ready,
        model_version=model_service.sklearn_version,
    )


@router.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(req: PredictionRequest) -> PredictionResponse:
    if not model_service.is_ready:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    try:
        price = model_service.predict(req)
    except Exception as exc:  # noqa: BLE001 — surface any inference error as 500
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(
        predicted_price=round(price, 2),
        predicted_price_formatted=format_inr(price),
    )
