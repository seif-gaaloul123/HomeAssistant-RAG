import re

FRONT_MATTER = re.compile(r"\A---\s*\n.*?\n---\s*(?:\n|$)", re.S)
RAW_BLOCK = re.compile(r"{%-?\s*raw\s*-?%}(.*?){%-?\s*endraw\s*-?%}", re.S)
COMMENT_BLOCK = re.compile(r"{%-?\s*comment\s*-?%}.*?{%-?\s*endcomment\s*-?%}", re.S)
# {% term "configuration.yaml" %} -> configuration.yaml
TERM_TAG = re.compile(r"{%-?\s*term\s+(?:\"([^\"]+)\"|'([^']+)'|([^\s%|]+))[^%]*?-?%}")
LIQUID_TAG = re.compile(r"{%.*?%}|{{.*?}}", re.S)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
FENCE = re.compile(r"^\s*(```|~~~)")

# only real html tags, so things like <your_ip> stay
TAG_NAMES = ("a|abbr|b|blockquote|br|center|code|details|div|em|figcaption|figure|font|"
             "h[1-6]|hr|i|iframe|img|li|ol|p|pre|section|small|source|span|strong|sub|"
             "summary|sup|table|tbody|td|tfoot|th|thead|tr|u|ul|video")
HTML_TAG = re.compile(rf"</?(?:{TAG_NAMES})(?:\s[^>]*)?/?>", re.I)

# braces inside {% raw %} get swapped for these so the liquid pass leaves them alone
OPEN, CLOSE = "\ue000", "\ue001"


def split_front_matter(text):
    m = FRONT_MATTER.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(0).splitlines()[1:-1]:
        kv = re.match(r"^([A-Za-z0-9_-]+)\s*:\s*(.*?)\s*$", line)
        # indented keys (related: > title:) don't match, and the first one wins
        if kv and kv.group(1) not in meta:
            meta[kv.group(1)] = kv.group(2).strip().strip("'\"")
    return meta, text[m.end():]


def protect_raw(m):
    return m.group(1).replace("{", OPEN).replace("}", CLOSE)


def clean_prose(text):
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"^\s*\[[^\]]+\]:\s+\S+.*$", "", text, flags=re.M)
    text = re.sub(r"<https?://[^>]+>", "", text)
    text = HTML_TAG.sub("", text)
    return text.replace("&nbsp;", " ")


def prose_only(text, fn):
    # run fn on prose, leave fenced code alone
    out, prose = [], []
    in_code = False

    def flush():
        if prose:
            out.extend(fn("\n".join(prose)).split("\n"))
            prose.clear()

    for line in text.split("\n"):
        if FENCE.match(line):
            flush()
            in_code = not in_code
            out.append(line)
        elif in_code:
            out.append(line)
        else:
            prose.append(line)
    flush()
    return "\n".join(out)


def clean_markdown(text):
    text = text.replace("\r\n", "\n")
    meta, body = split_front_matter(text)

    body = HTML_COMMENT.sub("", body)
    body = RAW_BLOCK.sub(protect_raw, body)
    body = COMMENT_BLOCK.sub("", body)
    body = TERM_TAG.sub(lambda m: m.group(1) or m.group(2) or m.group(3), body)
    body = LIQUID_TAG.sub("", body)  # only the tags, the text between them stays
    body = prose_only(body, clean_prose)
    body = body.replace(OPEN, "{").replace(CLOSE, "}")

    # trailing spaces only, leading ones are yaml indentation
    body = "\n".join(line.rstrip() for line in body.split("\n"))
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    title = meta.get("title") or ""
    if title and not re.search(r"^\s*#\s+" + re.escape(title), body, re.M | re.I):
        desc = meta.get("description", "")
        body = f"# {title}\n\n" + (f"{desc}\n\n" if desc else "") + body
    return meta, body.strip()
