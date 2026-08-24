import type { PredictionRequest, PredictionResponse } from "../types/prediction";

// Never hard-code the URL in components — it comes from the environment.
export const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function predictPrice(
  payload: PredictionRequest,
): Promise<PredictionResponse> {
  let res: Response;
  try {
    res = await fetch(`${BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new Error(
      `Could not reach the API at ${BASE_URL}. Is the backend running?`,
    );
  }

  if (!res.ok) {
    let detail = `Request failed with status ${res.status}.`;
    try {
      const data = await res.json();
      if (data?.detail) {
        detail =
          typeof data.detail === "string"
            ? data.detail
            : "Please check your inputs and try again.";
      }
    } catch {
      /* response had no JSON body */
    }
    throw new Error(detail);
  }

  return (await res.json()) as PredictionResponse;
}
