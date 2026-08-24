"""API tests: health check, a happy-path prediction, and validation (422)."""

VALID_PAYLOAD = {
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


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_happy_path(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_price"] > 0
    assert body["currency"] == "INR"
    assert body["predicted_price_formatted"].startswith("₹")


def test_predict_unknown_location_maps_to_other(client):
    # An unseen location must not crash — it is grouped into "other".
    payload = {**VALID_PAYLOAD, "location": "atlantis-not-a-real-city"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["predicted_price"] > 0


def test_predict_missing_field_returns_422(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "area_sqft"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_invalid_type_returns_422(client):
    payload = {**VALID_PAYLOAD, "area_sqft": "not-a-number"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_nonpositive_area_returns_422(client):
    payload = {**VALID_PAYLOAD, "area_sqft": 0}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
