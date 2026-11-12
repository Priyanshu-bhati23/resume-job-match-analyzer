"""Skill extraction against a curated vocabulary.

Exact vocabulary matching is deliberate: it is explainable ("the posting says
Kubernetes, your resume does not") and cannot hallucinate a skill. Aliases map
spelling variants onto one canonical name.
"""

from __future__ import annotations

import re

# canonical name -> aliases (lower-case). The canonical name is matched too.
SKILLS: dict[str, tuple[str, ...]] = {
    # languages
    "Python": (),
    "JavaScript": ("js",),
    "TypeScript": ("ts",),
    "Java": (),
    "Go": ("golang",),
    "Rust": (),
    "C++": ("cpp",),
    "C#": ("csharp",),
    "SQL": (),
    "Bash": ("shell scripting",),
    "Kotlin": (),
    "Swift": (),
    # AI / ML
    "LLM": ("llms", "large language model", "large language models"),
    "RAG": ("retrieval augmented generation", "retrieval-augmented generation"),
    "Agents": ("agent", "agentic", "ai agents"),
    "LangChain": (),
    "LangGraph": (),
    "LlamaIndex": (),
    "Prompt engineering": ("prompting", "prompt design"),
    "Embeddings": ("embedding",),
    "Vector search": ("vector database", "vector databases", "vector db", "semantic search"),
    "FAISS": (),
    "ChromaDB": ("chroma",),
    "Pinecone": (),
    "Weaviate": (),
    "pgvector": (),
    "Fine-tuning": ("finetuning", "fine tuning", "lora"),
    "Evaluation": ("evals", "llm evaluation"),
    "PyTorch": (),
    "TensorFlow": (),
    "scikit-learn": ("sklearn",),
    "Hugging Face": ("huggingface", "transformers"),
    "NLP": ("natural language processing",),
    "Computer vision": (),
    "MLOps": (),
    "OpenAI API": ("openai",),
    "Anthropic API": ("anthropic", "claude"),
    "MCP": ("model context protocol",),
    "Tool calling": ("tool use", "function calling"),
    # backend / web
    "FastAPI": (),
    "Django": (),
    "Flask": (),
    "Node.js": ("nodejs", "node"),
    "Express": ("express.js",),
    "React": ("react.js", "reactjs"),
    "Next.js": ("nextjs",),
    "REST APIs": ("rest api", "rest", "restful"),
    "GraphQL": (),
    "gRPC": (),
    "WebSockets": ("websocket",),
    "Microservices": (),
    # data
    "PostgreSQL": ("postgres",),
    "MySQL": (),
    "MongoDB": ("mongo",),
    "Redis": (),
    "Kafka": (),
    "Elasticsearch": (),
    "Spark": ("pyspark",),
    "Airflow": (),
    "dbt": (),
    "Pandas": (),
    "NumPy": (),
    # devops / cloud
    "Docker": ("containerization", "containers"),
    "Kubernetes": ("k8s",),
    "CI/CD": ("ci cd", "continuous integration", "continuous delivery"),
    "GitHub Actions": (),
    "Jenkins": (),
    "Terraform": (),
    "AWS": ("amazon web services",),
    "GCP": ("google cloud",),
    "Azure": (),
    "Linux": (),
    "Nginx": (),
    "Git": (),
    "Prometheus": (),
    "Grafana": (),
    "Observability": ("monitoring",),
    "Web scraping": ("scraping",),
    "Testing": ("pytest", "unit tests", "unit testing"),
}


def _pattern(term: str) -> str:
    # Word boundaries that also work for terms ending in symbols ("C++", "CI/CD", "Node.js").
    return rf"(?<![A-Za-z0-9+#]){re.escape(term)}(?![A-Za-z0-9+#])"


_COMPILED = {
    name: re.compile("|".join(_pattern(t) for t in (name.lower(), *aliases)), re.I) for name, aliases in SKILLS.items()
}
# Short or ambiguous names must match case-sensitively to avoid false hits ("go to market", "rest of").
_CASE_SENSITIVE = {
    "Go": re.compile(r"(?<![A-Za-z0-9])(?:Go|Golang|golang)(?![A-Za-z0-9])"),
    "Rust": re.compile(r"(?<![A-Za-z0-9])Rust(?![A-Za-z0-9])"),
    "Swift": re.compile(r"(?<![A-Za-z0-9])Swift(?![A-Za-z0-9])"),
    "Spark": re.compile(r"(?<![A-Za-z0-9])(?:Spark|PySpark)(?![A-Za-z0-9])"),
    "Express": re.compile(r"(?<![A-Za-z0-9])Express(?:\.js)?(?![A-Za-z0-9])"),
    "REST APIs": re.compile(r"(?<![A-Za-z0-9])(?:REST(?:ful)?(?: APIs?)?|rest apis?)(?![A-Za-z0-9])"),
    "Agents": re.compile(r"(?<![A-Za-z0-9])(?:[Aa]gents|[Aa]gentic|AI [Aa]gents?)(?![A-Za-z0-9])"),
    "Testing": re.compile(r"(?<![A-Za-z0-9])(?:pytest|[Uu]nit test(?:s|ing)?|[Tt]est suite|[Tt]esting)(?![A-Za-z0-9])"),
}
_COMPILED.update(_CASE_SENSITIVE)


def extract_skills(text: str) -> list[str]:
    """Canonical skill names found in `text`, in vocabulary order."""
    text = " ".join(text.split())  # PDF extraction can break "vector\nsearch" across lines
    return [name for name, pattern in _COMPILED.items() if pattern.search(text)]
