// Dropdown options, sourced from the model's exported schema so the frontend and
// the trained model always agree on the allowed categories.
import schema from "./model_columns.json";

export const LOCATIONS: string[] = schema.categorical_options.location_grouped;
export const FURNISHING: string[] = schema.categorical_options.Furnishing;
export const TRANSACTION: string[] = schema.categorical_options.Transaction;
export const OWNERSHIP: string[] = schema.categorical_options.Ownership;
export const FACING: string[] = schema.categorical_options.facing;

// Pretty-print a location slug like "navi-mumbai" -> "Navi Mumbai".
export function prettyLocation(slug: string): string {
  return slug
    .split("-")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}
