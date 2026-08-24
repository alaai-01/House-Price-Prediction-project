import { useState } from "react";
import type { FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import type { PredictionRequest } from "../types/prediction";
import { predictPrice } from "../api/predictionClient";
import {
  LOCATIONS,
  FURNISHING,
  TRANSACTION,
  OWNERSHIP,
  FACING,
  prettyLocation,
} from "../data/formOptions";

// The form keeps every field as a string (that is what <input>/<select> give us)
// and converts to numbers only at submit time.
type FormState = Record<keyof PredictionRequest, string>;

const INITIAL: FormState = {
  location: "mumbai",
  area_sqft: "1000",
  bhk: "2",
  bathroom: "2",
  balcony: "1",
  car_parking: "1",
  floor_num: "3",
  furnishing: "Semi-Furnished",
  transaction: "Resale",
  ownership: "Freehold",
  facing: "East",
};

// [field, label, min] for the numeric inputs.
const NUMERIC_FIELDS: Array<[keyof PredictionRequest, string, number]> = [
  ["area_sqft", "Area (sq ft)", 1],
  ["bhk", "Bedrooms (BHK)", 0],
  ["bathroom", "Bathrooms", 0],
  ["balcony", "Balconies", 0],
  ["car_parking", "Car parking", 0],
  ["floor_num", "Floor number", -5],
];

export default function PredictionForm() {
  const navigate = useNavigate();
  const [form, setForm] = useState<FormState>(INITIAL);
  const [errors, setErrors] = useState<Partial<Record<keyof PredictionRequest, string>>>({});
  const [apiError, setApiError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  function update(field: keyof PredictionRequest, value: string) {
    setForm((f) => ({ ...f, [field]: value }));
    setErrors((e) => ({ ...e, [field]: undefined }));
  }

  function validate(): PredictionRequest | null {
    const next: Partial<Record<keyof PredictionRequest, string>> = {};

    const area = Number(form.area_sqft);
    if (!form.area_sqft.trim() || Number.isNaN(area) || area <= 0) {
      next.area_sqft = "Enter an area greater than 0.";
    }

    const intFields: Array<[keyof PredictionRequest, number]> = [
      ["bhk", 0],
      ["bathroom", 0],
      ["balcony", 0],
      ["car_parking", 0],
      ["floor_num", -5],
    ];
    for (const [field, min] of intFields) {
      const v = Number(form[field]);
      if (form[field].trim() === "" || Number.isNaN(v) || !Number.isInteger(v) || v < min) {
        next[field] = `Enter a whole number ≥ ${min}.`;
      }
    }

    setErrors(next);
    if (Object.keys(next).length > 0) return null;

    return {
      area_sqft: area,
      bhk: Number(form.bhk),
      bathroom: Number(form.bathroom),
      balcony: Number(form.balcony),
      car_parking: Number(form.car_parking),
      floor_num: Number(form.floor_num),
      location: form.location,
      furnishing: form.furnishing,
      transaction: form.transaction,
      ownership: form.ownership,
      facing: form.facing,
    };
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setApiError(null);
    const payload = validate();
    if (!payload) return;

    setLoading(true);
    try {
      const result = await predictPrice(payload);
      navigate("/result", { state: { input: payload, result } });
    } catch (err) {
      setApiError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form className="form" onSubmit={handleSubmit} noValidate>
      <div className="grid">
        {/* Location */}
        <label className="field field--wide">
          <span>Location</span>
          <select value={form.location} onChange={(e) => update("location", e.target.value)}>
            {LOCATIONS.map((loc) => (
              <option key={loc} value={loc}>
                {prettyLocation(loc)}
              </option>
            ))}
          </select>
        </label>

        {/* Numeric inputs */}
        {NUMERIC_FIELDS.map(([field, label, min]) => (
          <label className="field" key={field}>
            <span>{label}</span>
            <input
              type="number"
              inputMode="numeric"
              min={min}
              step={field === "area_sqft" ? "any" : 1}
              value={form[field]}
              onChange={(e) => update(field, e.target.value)}
              aria-invalid={Boolean(errors[field])}
            />
            {errors[field] && <small className="error">{errors[field]}</small>}
          </label>
        ))}

        {/* Categorical selects */}
        <label className="field">
          <span>Furnishing</span>
          <select value={form.furnishing} onChange={(e) => update("furnishing", e.target.value)}>
            {FURNISHING.map((o) => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
        </label>

        <label className="field">
          <span>Transaction</span>
          <select value={form.transaction} onChange={(e) => update("transaction", e.target.value)}>
            {TRANSACTION.map((o) => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
        </label>

        <label className="field">
          <span>Ownership</span>
          <select value={form.ownership} onChange={(e) => update("ownership", e.target.value)}>
            {OWNERSHIP.map((o) => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
        </label>

        <label className="field">
          <span>Facing</span>
          <select value={form.facing} onChange={(e) => update("facing", e.target.value)}>
            {FACING.map((o) => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
        </label>
      </div>

      {apiError && <p className="api-error" role="alert">{apiError}</p>}

      <button className="submit" type="submit" disabled={loading}>
        {loading ? "Predicting…" : "Predict price"}
      </button>
    </form>
  );
}
