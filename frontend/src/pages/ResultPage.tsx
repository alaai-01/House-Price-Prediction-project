import { Link, useLocation, Navigate } from "react-router-dom";
import type { PredictionRequest, PredictionResponse } from "../types/prediction";
import { prettyLocation } from "../data/formOptions";

interface ResultState {
  input: PredictionRequest;
  result: PredictionResponse;
}

export default function ResultPage() {
  const location = useLocation();
  const state = location.state as ResultState | null;

  // Deep-linked or refreshed without state -> send the user back home.
  if (!state?.result) {
    return <Navigate to="/" replace />;
  }

  const { input, result } = state;

  const summary: Array<[string, string]> = [
    ["Location", prettyLocation(input.location)],
    ["Area", `${input.area_sqft.toLocaleString()} sq ft`],
    ["Configuration", `${input.bhk} BHK · ${input.bathroom} bath · ${input.balcony} balcony`],
    ["Floor", String(input.floor_num)],
    ["Car parking", String(input.car_parking)],
    ["Furnishing", input.furnishing],
    ["Transaction", input.transaction],
    ["Ownership", input.ownership],
    ["Facing", input.facing],
  ];

  return (
    <main className="page">
      <section className="card result-card">
        <p className="result-label">Estimated price</p>
        <p className="result-price">{result.predicted_price_formatted}</p>
        <p className="result-exact">
          ≈ ₹ {result.predicted_price.toLocaleString("en-IN")}
        </p>

        <dl className="summary">
          {summary.map(([k, v]) => (
            <div className="summary-row" key={k}>
              <dt>{k}</dt>
              <dd>{v}</dd>
            </div>
          ))}
        </dl>

        <Link className="submit link-btn" to="/">
          ← Predict another
        </Link>
      </section>
    </main>
  );
}
