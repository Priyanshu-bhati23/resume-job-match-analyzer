export type Status = "strong" | "partial" | "missing";

export interface RequirementMatch {
  requirement: string;
  score: number;
  evidence: string | null;
  status: Status;
}

export interface Suggestion {
  category: "missing_skill" | "weak_evidence" | "wording" | "structure";
  priority: "high" | "medium" | "low";
  title: string;
  detail: string;
  resume_excerpt: string | null;
  rewrite: string | null;
}

export interface Analysis {
  score: { overall: number; semantic: number; skills: number | null; weights: Record<string, number> };
  skills: { matched: string[]; missing: string[]; extra: string[]; coverage: number | null };
  requirements: RequirementMatch[];
  summary: string;
  suggestions: Suggestion[];
  meta: { embedder: string; suggestions_by: string; resume_chars: number };
}

export interface AnalyzeInput {
  jobDescription: string;
  resumeText?: string;
  resumeFile?: File | null;
}

export async function analyze(input: AnalyzeInput, signal?: AbortSignal): Promise<Analysis> {
  const form = new FormData();
  form.append("job_description", input.jobDescription);
  if (input.resumeFile) form.append("resume_file", input.resumeFile);
  else if (input.resumeText) form.append("resume_text", input.resumeText);

  let response: Response;
  try {
    response = await fetch("/api/analyze", { method: "POST", body: form, signal });
  } catch (error) {
    if ((error as Error).name === "AbortError") throw error;
    throw new Error("Could not reach the API. Is the backend running on port 8000?");
  }
  if (!response.ok) {
    // FastAPI returns {"detail": "..."} for our errors and a list for validation errors.
    const body = await response.json().catch(() => null);
    const detail = typeof body?.detail === "string" ? body.detail : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return response.json();
}
