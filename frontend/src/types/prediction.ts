// Mirrors the FastAPI backend schema (backend/app/schemas/prediction.py).

export interface PredictionRequest {
  area_sqft: number;
  bhk: number;
  bathroom: number;
  balcony: number;
  car_parking: number;
  floor_num: number;
  location: string;
  furnishing: string;
  transaction: string;
  ownership: string;
  facing: string;
}

export interface PredictionResponse {
  predicted_price: number;
  predicted_price_formatted: string;
  currency: string;
}
