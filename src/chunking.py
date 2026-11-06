import hashlib
import re

from .config import MAX_CHARS, OVERLAP

FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
MIN_CHUNK = 40


def parse_blocks(text):
    """heading / code / para blocks. A fenced code block is one block, and
    '# comments' inside it are not headings."""
    blocks, stack, buf = [], [], []
    in_code, fence = False, ""

    def section():
        return " > ".join(title for _, title in stack)

    def flush(kind="para"):
        content = "\n".join(buf).strip("\n")
        if content.strip():
            blocks.append({"type": kind, "text": content, "section": section()})
        buf.clear()

    for line in text.split("\n"):
        m = FENCE.match(line)
        if in_code:
            buf.append(line)
            if m and m.group(1) == fence:
                flush("code")
                in_code = False
            continue
        if m:
            flush()
            in_code, fence = True, m.group(1)
            buf.append(line)
            continue
        h = HEADING.match(line)
        if h:
            flush()
            level, title = len(h.group(1)), h.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            blocks.append({"type": "heading", "text": line.strip(),
                           "section": section(), "level": level})
            continue
        if not line.strip():
            flush()
            continue
        buf.append(line)
    flush("code" if in_code else "para")
    return blocks


def split_long(text, max_chars):
    if len(text) <= max_chars:
        return [text]
    out, cur = [], ""
    for line in text.split("\n"):
        while len(line) > max_chars:
            if cur:
                out.append(cur)
                cur = ""
            out.append(line[:max_chars])
            line = line[max_chars:]
        if cur and len(cur) + len(line) + 1 > max_chars:
            out.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        out.append(cur)
    return out


def chunk_id(source, idx, text):
    return hashlib.sha1(f"{source}:{idx}:{text[:100]}".encode()).hexdigest()[:18]


def make_chunk(doc, idx, text, strategy, section=""):
    parts = [p for p in section.split(" > ") if p]
    if parts and parts[0].strip().lower() == doc["title"].strip().lower():
        parts = parts[1:]  # page title is already in the header
    header = doc["title"]
    if parts:
        header += " > " + " > ".join(parts)
    return {
        "chunk_id": chunk_id(doc["source"], idx, text),
        "doc_id": doc["doc_id"],
        "source": doc["source"],
        "title": doc["title"],
        "section": " > ".join(parts),
        "strategy": strategy,
        "text": text,
        "embed_text": f"{header}\n\n{text}",  # this is what gets embedded / bm25'd / reranked
    }


def finalize(doc, items, strategy):
    items = [(s, t) for s, t in items if t.strip()]
    kept = [(s, t) for s, t in items if len(t.strip()) >= MIN_CHUNK] or items[:1]
    return [make_chunk(doc, i, t, strategy, s) for i, (s, t) in enumerate(kept)]


# --- regular: ignores structure, just packs paragraphs up to the size limit ---

def regular_chunks(doc, chunk_size=MAX_CHARS, overlap=OVERLAP):
    paras = [p.strip() for p in re.split(r"\n\s*\n", doc["content"]) if p.strip()]
    pieces, current = [], ""
    for para in paras:
        if len(para) > chunk_size:
            if current:
                pieces.append(current)
                current = ""
            start = 0
            while start < len(para):
                pieces.append(para[start:start + chunk_size])
                start += max(1, chunk_size - overlap)
            continue
        candidate = (current + "\n\n" + para).strip()
        if len(candidate) > chunk_size and current:
            pieces.append(current)
            tail = current[-overlap:] if overlap else ""
            current = (tail + "\n\n" + para).strip()
        else:
            current = candidate
    if current:
        pieces.append(current)
    return finalize(doc, [("", p) for p in pieces], "regular")


# --- semantic: cut where neighbouring sentences stop being similar ---

def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n(?=\s*(?:[-*+]|\d+\.)\s)", text)
    return [s.strip() for s in parts if s and s.strip()]


def semantic_chunks(doc, model, max_chars=MAX_CHARS, min_chars=250, threshold=0.45):
    units, heading = [], ""
    for b in parse_blocks(doc["content"]):
        if b["type"] == "heading":
            heading = f"{heading}\n{b['text']}".strip()
            continue
        if b["type"] == "code":
            pieces = split_long(b["text"], max_chars)  # keep code in one piece
        else:
            pieces = sentences(b["text"])
        for p in pieces:
            if heading:
                p, heading = f"{heading}\n{p}", ""
            units.append(p)
    if heading:
        units.append(heading)

    if not units:
        return [dict(c, strategy="semantic") for c in regular_chunks(doc)]
    if len(units) == 1:
        return finalize(doc, [("", units[0])], "semantic")

    vecs = model.encode(units, normalize_embeddings=True, show_progress_bar=False)
    groups, cur, cur_len = [], [units[0]], len(units[0])
    for i in range(1, len(units)):
        sim = float((vecs[i - 1] * vecs[i]).sum())
        too_big = cur_len + len(units[i]) + 1 > max_chars
        topic_changed = sim < threshold and cur_len >= min_chars
        if too_big or topic_changed:
            groups.append("\n".join(cur))
            cur, cur_len = [units[i]], len(units[i])
        else:
            cur.append(units[i])
            cur_len += len(units[i]) + 1
    groups.append("\n".join(cur))

    merged = []
    for g in groups:
        if merged and len(g) < min_chars // 2 and len(merged[-1]) + len(g) + 1 <= max_chars:
            merged[-1] += "\n" + g
        else:
            merged.append(g)
    return finalize(doc, [("", g) for g in merged], "semantic")


# --- heading: one chunk per section (h1-h3), code never cut ---

def heading_chunks(doc, max_chars=MAX_CHARS, min_chars=200, split_level=3):
    sections, cur = [], None
    for b in parse_blocks(doc["content"]):
        if b["type"] == "heading" and b["level"] <= split_level:
            cur = {"section": b["section"], "parts": [b["text"]]}
            sections.append(cur)
            continue
        if cur is None:
            cur = {"section": b["section"], "parts": []}
            sections.append(cur)
        cur["parts"].append(b["text"])

    packed = []
    for sec in sections:
        buf = ""
        for part in (p for t in sec["parts"] for p in split_long(t, max_chars)):
            if buf and len(buf) + len(part) + 2 > max_chars:
                packed.append((sec["section"], buf))
                buf = part
            else:
                buf = f"{buf}\n\n{part}" if buf else part
        if buf:
            packed.append((sec["section"], buf))

    # glue tiny chunks onto the next one if it still fits (label of the later one wins)
    merged = []
    for sec, text in packed:
        if merged and len(merged[-1][1]) < min_chars and len(merged[-1][1]) + len(text) + 2 <= max_chars:
            merged[-1] = (sec, merged[-1][1] + "\n\n" + text)
        else:
            merged.append((sec, text))
    return finalize(doc, merged, "heading")
