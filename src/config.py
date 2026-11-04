import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
INDEXES = DATA / "indexes"
EVAL = DATA / "eval"
REPORTS = ROOT / "reports"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

STRATEGIES = ("regular", "semantic", "heading")

# same budget for all three strategies. MiniLM cuts at 256 tokens (~1000 chars)
MAX_CHARS = 900
OVERLAP = 150
