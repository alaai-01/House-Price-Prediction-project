"""Turn an API request into the exact one-row DataFrame the Pipeline was trained on.

The training columns (from notebooks/model_columns.json) are:
  numeric:     area_sqft, floor_num, bathroom, balcony, car_parking, bhk
  categorical: location_grouped, Furnishing, Transaction, Ownership, facing

The exported model is a full sklearn Pipeline, so no manual encoding/scaling is done
here — we only need to reproduce the column names and map the location the same way
the notebook did (top-50 locations kept, everything else -> "other").
"""
import pandas as pd

from app.schemas.prediction import PredictionRequest


def build_feature_frame(req: PredictionRequest, known_locations: set[str]) -> pd.DataFrame:
    loc = req.location.strip().lower()
    location_grouped = loc if loc in known_locations else "other"

    row = {
        # numeric
        "area_sqft": req.area_sqft,
        "floor_num": req.floor_num,
        "bathroom": req.bathroom,
        "balcony": req.balcony,
        "car_parking": req.car_parking,
        "bhk": req.bhk,
        # categorical (column names match the training frame exactly)
        "location_grouped": location_grouped,
        "Furnishing": req.furnishing,
        "Transaction": req.transaction,
        "Ownership": req.ownership,
        "facing": req.facing,
    }
    return pd.DataFrame([row])
