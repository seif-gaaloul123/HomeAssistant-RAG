import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from .cleaning import clean_markdown
from .cli import check, get
from .config import PROCESSED

EXTENSIONS = {".md", ".markdown", ".mdown"}


def git_sha(repo):
    try:
        out = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                      text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return None


def main():
    check("repo", "limit")
    repo_arg = get("repo")
    if not repo_arg:
        raise SystemExit("Usage: python -m src.ingest --repo ../home-assistant.io [--limit 300]")
    limit = get("limit", 0)  # only for quick smoke tests

    repo = Path(repo_arg).expanduser().resolve()
    roots = [repo / "source" / "_integrations", repo / "source" / "_docs"]
    missing = [str(r) for r in roots if not r.exists()]
    if missing:
        raise SystemExit("Missing expected source folders: " + ", ".join(missing))

    paths = sorted(f for root in roots for f in root.rglob("*")
                   if f.is_file() and f.suffix.lower() in EXTENSIONS)
    if limit:
        paths = paths[:limit]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / "documents.jsonl"
    count = 0
    with out_path.open("w", encoding="utf-8") as out:
        for path in tqdm(paths, desc="Cleaning documents"):
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
                meta, text = clean_markdown(raw)
                if len(text) < 80:
                    continue
                rel = path.relative_to(repo).as_posix()
                doc = {
                    "doc_id": hashlib.sha1(rel.encode()).hexdigest()[:16],
                    "source": rel,
                    "collection": path.parent.name,
                    "title": meta.get("title", path.stem.replace("_", " ")),
                    "content": text,
                }
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                count += 1
            except Exception as exc:
                print(f"[WARN] {path}: {exc}")

    manifest = {
        "repo_commit": git_sha(repo),
        "documents": count,
        "limit": limit or None,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    (PROCESSED / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {count} cleaned documents to {out_path} (commit {manifest['repo_commit']})")


if __name__ == "__main__":
    main()
