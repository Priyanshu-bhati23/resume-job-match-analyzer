import { useRef, useState } from "react";
import { analyze, type Analysis, type RequirementMatch, type Suggestion } from "./api";
import { SAMPLE_JOB, SAMPLE_RESUME } from "./sample";

type ResumeMode = "upload" | "paste";

export default function App() {
  const [mode, setMode] = useState<ResumeMode>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [resumeText, setResumeText] = useState("");
  const [job, setJob] = useState("");
  const [result, setResult] = useState<Analysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const inFlight = useRef<AbortController | null>(null);

  const hasResume = mode === "upload" ? file !== null : resumeText.trim().length > 0;
  const canSubmit = hasResume && job.trim().length > 0 && !loading;

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!canSubmit) return;
    inFlight.current?.abort(); // a newer request supersedes an older one
    const controller = new AbortController();
    inFlight.current = controller;
    setLoading(true);
    setError(null);
    try {
      const analysis = await analyze(
        mode === "upload" ? { jobDescription: job, resumeFile: file } : { jobDescription: job, resumeText },
        controller.signal,
      );
      setResult(analysis);
    } catch (err) {
      if ((err as Error).name !== "AbortError") setError((err as Error).message);
    } finally {
      if (inFlight.current === controller) setLoading(false);
    }
  }

  function loadSample() {
    setMode("paste");
    setResumeText(SAMPLE_RESUME);
    setJob(SAMPLE_JOB);
    setFile(null);
  }

  return (
    <div className="page">
      <header className="masthead">
        <h1>Resume &amp; Job-Match Analyzer</h1>
        <p>Scores how well a resume covers a job posting, line by line, and says what to change.</p>
      </header>

      <main className="layout">
        <form className="panel inputs" onSubmit={submit}>
          <div className="field">
            <div className="field-head">
              <label htmlFor={mode === "upload" ? "resume-file" : "resume-text"}>Resume</label>
              <div className="tabs" role="tablist" aria-label="Resume input">
                {(["upload", "paste"] as const).map((m) => (
                  <button
                    key={m}
                    type="button"
                    role="tab"
                    aria-selected={mode === m}
                    className={mode === m ? "tab active" : "tab"}
                    onClick={() => setMode(m)}
                  >
                    {m === "upload" ? "Upload" : "Paste text"}
                  </button>
                ))}
              </div>
            </div>
            {mode === "upload" ? (
              <label className={file ? "drop has-file" : "drop"}>
                <input
                  id="resume-file"
                  type="file"
                  accept=".pdf,.txt,.md,application/pdf,text/plain"
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                />
                <span className="drop-title">{file ? file.name : "Choose a PDF, .txt or .md"}</span>
                <span className="drop-hint">
                  {file ? `${Math.max(1, Math.round(file.size / 1024))} KB · click to replace` : "Up to 5 MB"}
                </span>
              </label>
            ) : (
              <textarea
                id="resume-text"
                rows={10}
                value={resumeText}
                onChange={(e) => setResumeText(e.target.value)}
                placeholder="Paste the resume text"
              />
            )}
          </div>

          <div className="field">
            <div className="field-head">
              <label htmlFor="job">Job description</label>
            </div>
            <textarea
              id="job"
              rows={12}
              value={job}
              onChange={(e) => setJob(e.target.value)}
              placeholder="Paste the full posting, including the requirements list"
            />
          </div>

          <div className="actions">
            <button type="submit" className="primary" disabled={!canSubmit}>
              {loading ? "Analysing…" : "Analyse fit"}
            </button>
            <button type="button" className="link" onClick={loadSample}>
              Load an example
            </button>
          </div>
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
        </form>

        <section className="results" aria-live="polite" aria-busy={loading}>
          {result ? <Results data={result} stale={loading} /> : <EmptyState loading={loading} />}
        </section>
      </main>
    </div>
  );
}

function EmptyState({ loading }: { loading: boolean }) {
  return (
    <div className="panel empty">
      <h2>{loading ? "Reading both documents…" : "Nothing analysed yet"}</h2>
      <p>
        Add a resume and a job description. You get an overall fit score, which requirements the resume backs up
        with evidence, the skills the posting names that the resume does not, and specific edits.
      </p>
    </div>
  );
}

function Results({ data, stale }: { data: Analysis; stale: boolean }) {
  const { score, skills, requirements, suggestions, meta } = data;
  const counts = {
    strong: requirements.filter((r) => r.status === "strong").length,
    partial: requirements.filter((r) => r.status === "partial").length,
    missing: requirements.filter((r) => r.status === "missing").length,
  };

  return (
    <div className={stale ? "stack stale" : "stack"}>
      <div className="panel score-panel">
        <Dial value={score.overall} />
        <div className="score-detail">
          <p className="summary">{data.summary}</p>
          <Bar
            label="Requirement coverage"
            value={score.semantic}
            note={`${Math.round(score.weights.semantic * 100)}% of score`}
          />
          {score.skills === null ? (
            <p className="muted small">The posting names no skills from the vocabulary, so skills are not scored.</p>
          ) : (
            <Bar
              label="Named skills present"
              value={score.skills}
              note={`${Math.round(score.weights.skills * 100)}% of score`}
            />
          )}
        </div>
      </div>

      <div className="panel">
        <h2>Skills</h2>
        <ChipRow title="Missing from the resume" items={skills.missing} tone="missing" empty="None — every named skill is present." />
        <ChipRow title="Matched" items={skills.matched} tone="strong" empty="No named skills matched." />
        {skills.extra.length > 0 && <ChipRow title="On the resume, not asked for" items={skills.extra} tone="neutral" />}
      </div>

      <div className="panel">
        <h2>
          Requirements{" "}
          <span className="tally">
            {counts.strong} strong · {counts.partial} partial · {counts.missing} missing
          </span>
        </h2>
        <ul className="requirements">
          {requirements.map((r) => (
            <Requirement key={r.requirement} item={r} />
          ))}
        </ul>
      </div>

      <div className="panel">
        <h2>Suggested edits</h2>
        {suggestions.length === 0 ? (
          <p className="muted">No changes suggested.</p>
        ) : (
          <ol className="suggestions">
            {suggestions.map((s, i) => (
              <SuggestionCard key={i} item={s} />
            ))}
          </ol>
        )}
      </div>

      <p className="meta">
        Embeddings: {meta.embedder} · Suggestions: {meta.suggestions_by === "rules" ? "rule-based" : meta.suggestions_by}{" "}
        · A screening heuristic, not a hiring prediction.
      </p>
    </div>
  );
}

function Dial({ value }: { value: number }) {
  const radius = 52;
  const circumference = 2 * Math.PI * radius;
  const tone = value >= 70 ? "strong" : value >= 40 ? "partial" : "missing";
  return (
    <div className="dial" role="img" aria-label={`Overall fit ${value} out of 100`}>
      <svg viewBox="0 0 120 120" width="132" height="132">
        <circle className="dial-track" cx="60" cy="60" r={radius} />
        <circle
          className={`dial-value ${tone}`}
          cx="60"
          cy="60"
          r={radius}
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - value / 100)}
          transform="rotate(-90 60 60)"
        />
      </svg>
      <div className="dial-label">
        <strong>{value}</strong>
        <span>overall fit</span>
      </div>
    </div>
  );
}

function Bar({ label, value, note }: { label: string; value: number; note: string }) {
  return (
    <div className="bar">
      <div className="bar-head">
        <span>{label}</span>
        <span className="bar-value">
          {value}
          <span className="muted small"> / 100 · {note}</span>
        </span>
      </div>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${value}%` }} />
      </div>
    </div>
  );
}

function ChipRow(props: { title: string; items: string[]; tone: "strong" | "missing" | "neutral"; empty?: string }) {
  return (
    <div className="chip-row">
      <h3>
        {props.title} <span className="count">{props.items.length}</span>
      </h3>
      {props.items.length === 0 ? (
        <p className="muted small">{props.empty}</p>
      ) : (
        <ul className="chips">
          {props.items.map((item) => (
            <li key={item} className={`chip ${props.tone}`}>
              {item}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

const STATUS_LABEL = { strong: "Strong", partial: "Partial", missing: "No evidence" } as const;

function Requirement({ item }: { item: RequirementMatch }) {
  return (
    <li className="requirement">
      <span className={`pill ${item.status}`}>{STATUS_LABEL[item.status]}</span>
      <div>
        <p className="requirement-text">{item.requirement}</p>
        {item.evidence ? (
          <p className="evidence">
            <span className="evidence-label">Closest resume line</span>
            {item.evidence}
          </p>
        ) : (
          <p className="muted small">Nothing on the resume comes close to this line.</p>
        )}
      </div>
    </li>
  );
}

const CATEGORY_LABEL = {
  missing_skill: "Missing skill",
  weak_evidence: "Weak evidence",
  wording: "Wording",
  structure: "Structure",
} as const;

function SuggestionCard({ item }: { item: Suggestion }) {
  return (
    <li className="suggestion">
      <div className="suggestion-head">
        <span className={`priority ${item.priority}`}>{item.priority}</span>
        <span className="category">{CATEGORY_LABEL[item.category]}</span>
      </div>
      <h3>{item.title}</h3>
      <p>{item.detail}</p>
      {item.resume_excerpt && (
        <div className="diff">
          <p className="diff-line before">
            <span>On your resume</span>
            {item.resume_excerpt}
          </p>
          {item.rewrite && (
            <p className="diff-line after">
              <span>Suggested</span>
              {item.rewrite}
            </p>
          )}
        </div>
      )}
    </li>
  );
}
