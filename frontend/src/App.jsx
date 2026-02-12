import { useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "/api";

export default function App() {
  const [jd, setJd] = useState("");
  const [file, setFile] = useState(null);
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("Idle");

  const canSubmit = useMemo(() => jd.trim().length > 0, [jd]);

  const onSubmit = async (event) => {
    event.preventDefault();
    setError("");
    setResult("");
    setLoading(true);
    setStatus("Submitting");

    const form = new FormData();
    form.append("jd", jd);
    if (file) {
      form.append("file", file);
    }

    try {
      const response = await fetch(`${API_BASE}/upload/`, {
        method: "POST",
        body: form,
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || `Request failed (${response.status})`);
      }

      const text = await response.text();
      setResult(text);
      setStatus("Complete");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
      setStatus("Failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">LLM Writes Job Applications</p>
          <h1>Craft a tailored application in minutes.</h1>
          <p className="subtitle">
            Upload your CV once, then paste job descriptions to generate
            cover-letter style responses.
          </p>
        </div>
        <div className="status-card">
          <p className="status-label">Session Status</p>
          <p className={`status-value ${loading ? "pulse" : ""}`}>{status}</p>
          <p className="status-hint">
            {file ? `CV loaded: ${file.name}` : "Upload a PDF CV to start."}
          </p>
        </div>
      </header>

      <main className="content">
        <form className="card" onSubmit={onSubmit}>
          <div className="field">
            <label htmlFor="cv">CV (PDF)</label>
            <input
              id="cv"
              type="file"
              accept="application/pdf"
              onChange={(event) => setFile(event.target.files?.[0] || null)}
            />
            <p className="help">
              Upload once and reuse it for multiple job descriptions.
            </p>
          </div>

          <div className="field">
            <label htmlFor="jd">Job description</label>
            <textarea
              id="jd"
              placeholder="Paste the role requirements, team context, and tech stack..."
              value={jd}
              onChange={(event) => setJd(event.target.value)}
              rows={10}
            />
          </div>

          <div className="actions">
            <button type="submit" disabled={!canSubmit || loading}>
              {loading ? "Generating..." : "Generate application"}
            </button>
            <button
              type="button"
              className="ghost"
              onClick={() => {
                setJd("");
                setResult("");
                setError("");
                setStatus("Idle");
              }}
            >
              Reset
            </button>
          </div>

          {error && <p className="error">{error}</p>}
        </form>

        <section className="card output">
          <div className="output-header">
            <h2>Generated response</h2>
            <span className="chip">Plain text</span>
          </div>
          {result ? (
            <pre>{result}</pre>
          ) : (
            <p className="empty">Your generated text will appear here.</p>
          )}
        </section>
      </main>
    </div>
  );
}
