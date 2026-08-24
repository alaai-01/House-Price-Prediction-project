import PredictionForm from "../components/PredictionForm";

export default function HomePage() {
  return (
    <main className="page">
      <header className="hero">
        <h1>🏠 House Price Predictor</h1>
        <p>
          Estimate the market price of an Indian residential property. Enter the
          details below and the model will predict a price.
        </p>
      </header>
      <section className="card">
        <PredictionForm />
      </section>
      <footer className="foot">
        Model: HistGradientBoosting (log target) · Test R² 0.875 · Trained on ~187k Kaggle listings
      </footer>
    </main>
  );
}
