import { Link } from "react-router-dom";

export default function NotFoundPage() {
  return (
    <main className="page">
      <section className="card" style={{ textAlign: "center" }}>
        <h1>404</h1>
        <p>That page doesn’t exist.</p>
        <Link className="submit link-btn" to="/">
          Go home
        </Link>
      </section>
    </main>
  );
}
