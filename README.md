# 🏠 House Price Prediction

An end-to-end machine-learning web application that predicts **Indian residential property prices** from listing features. It spans the full stack: a trained scikit-learn model, a FastAPI inference service, and a React + TypeScript frontend.

The model is a **HistGradientBoostingRegressor** trained on a log-transformed target over ~187k listings from the Kaggle [*House Price*](https://www.kaggle.com/datasets/juhibhojani/house-price) dataset.

| Metric (held-out test set) | Value |
| --- | --- |
| R² | **0.875** |
| MAE | ₹1.25 M |
| RMSE | ₹4.67 M |
| 5-fold CV R² | 0.858 ± 0.023 |

*(Linear Regression baseline: R² 0.631.)*

---

## Architecture

```
house_price/
├── notebooks/
│   ├── house_price_model.ipynb   # EDA → training → evaluation → model export
│   ├── house_price.pkl           # trained sklearn pipeline (committed, 1.8 MB)
│   ├── locations.json            # known locations for the UI
│   └── model_columns.json        # feature schema
├── backend/                      # FastAPI inference API
│   ├── app/
│   │   ├── main.py               # app factory + CORS + lifespan model load
│   │   ├── api/routes/           # prediction & health endpoints
│   │   ├── core/                 # settings/config
│   │   ├── schemas/              # Pydantic request/response models
│   │   ├── services/             # model loading & inference
│   │   └── utils/                # logging config
│   ├── models/                   # model artifacts served by the API
│   ├── tests/                    # pytest suite
│   ├── requirements.txt          # pinned deps (sklearn 1.6.1 — must match the pickle)
│   └── Dockerfile
├── frontend/                     # React + TypeScript + Vite UI
│   └── src/                      # pages (Home / Result), PredictionForm, API client
├── _build_notebook.py            # generates the notebook programmatically
└── House_Price_Prediction_Project_Guide.md
```

## The model contract

The exported `house_price.pkl` is a full sklearn `Pipeline` (impute + scale numeric, impute + one-hot categorical) wrapped in a `TransformedTargetRegressor(log1p / expm1)`, so **`predict()` returns prices directly in rupees** — no inverse transform needed downstream.

**Inputs (11 features):**

- **Numeric:** `area_sqft`, `floor_num`, `bathroom`, `balcony`, `car_parking`, `bhk`
- **Categorical:** `location`, `Furnishing`, `Transaction`, `Ownership`, `facing`

Unknown categories are handled gracefully (one-hot `handle_unknown="ignore"`); `location` is mapped to the top-50 training locations, otherwise `"other"`.

---

## Getting started

### 1. Dataset (optional — only needed to retrain)

The raw ~106 MB CSV is **not** committed. To re-run the notebook, download the [Kaggle dataset](https://www.kaggle.com/datasets/juhibhojani/house-price) and save it as `house_prices.csv` in the project root. The trained model is already committed, so the **API and frontend run without it**.

### 2. Backend (FastAPI)

> Requires Python with the pinned dependencies — **scikit-learn must be 1.6.1** to load the pickle.

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate      # Windows
# source .venv/bin/activate                          # macOS / Linux
pip install -r requirements.txt

cp .env.example .env        # adjust CORS origins if needed
uvicorn app.main:app --reload
```

API: <http://localhost:8000> · Interactive docs: <http://localhost:8000/docs> · Health: <http://localhost:8000/health>

Run the tests:

```bash
pytest
```

### 3. Frontend (React + Vite)

```bash
cd frontend
npm install
cp .env.example .env        # VITE_API_BASE_URL=http://localhost:8000
npm run dev
```

App: <http://localhost:5173>

---

## Tech stack

- **ML / Data:** Python, pandas, NumPy, scikit-learn, Jupyter
- **Backend:** FastAPI, Uvicorn, Pydantic, joblib
- **Frontend:** React 19, TypeScript, Vite, React Router

## License

Released for educational purposes. The dataset is © its original authors on Kaggle.
