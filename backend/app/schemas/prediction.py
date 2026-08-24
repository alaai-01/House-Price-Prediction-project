"""Request/response schemas for the prediction API.

The fields mirror exactly the features the trained scikit-learn Pipeline expects
(see notebooks/model_columns.json). Because the model is a full Pipeline wrapped in
a TransformedTargetRegressor, it takes raw feature values and returns rupees directly.
"""
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    # --- numeric features ---
    area_sqft: float = Field(..., gt=0, description="Carpet/super area in square feet.")
    bhk: int = Field(..., ge=0, le=20, description="Number of bedrooms (BHK).")
    bathroom: int = Field(..., ge=0, le=20)
    balcony: int = Field(..., ge=0, le=20)
    car_parking: int = Field(..., ge=0, le=20)
    floor_num: int = Field(..., ge=-5, le=200, description="Floor number; 0 = Ground, -1 = Basement.")

    # --- categorical features ---
    location: str = Field(..., min_length=1, description="City/locality; unknown values map to 'other'.")
    furnishing: str = Field(..., description="Furnished | Semi-Furnished | Unfurnished")
    transaction: str = Field(..., description="New Property | Resale | Rent/Lease | Other")
    ownership: str = Field(..., description="Freehold | Leasehold | Co-operative Society | Power Of Attorney")
    facing: str = Field(..., description="e.g. East, North - East, West")

    model_config = {
        "json_schema_extra": {
            "example": {
                "area_sqft": 1200,
                "bhk": 3,
                "bathroom": 2,
                "balcony": 2,
                "car_parking": 1,
                "floor_num": 5,
                "location": "mumbai",
                "furnishing": "Semi-Furnished",
                "transaction": "Resale",
                "ownership": "Freehold",
                "facing": "East",
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Predicted price in Indian rupees (₹).")
    predicted_price_formatted: str = Field(..., description="Human-friendly price, e.g. '₹ 1.25 Cr'.")
    currency: str = "INR"


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    model_version: str = Field(..., description="scikit-learn version the model was trained with.")
