import numpy as np
from src.chunking import parse_blocks, regular_chunks, heading_chunks, semantic_chunks

CODE = "```yaml\n# a comment, not a heading\nmqtt:\n  sensor:\n    - name: temp\n      state_topic: home/temp\n```"
CONTENT = (
    "# MQTT\n\nIntro text about the integration. " * 2
    + "\n\n## Setup\n\n" + ("Some setup prose that explains things. " * 12)
    + "\n\n" + CODE + "\n\n## Other\n\nShort section with enough text to be kept as a chunk here."
)
DOC = {"doc_id": "1", "source": "source/_integrations/mqtt.markdown", "title": "MQTT", "content": CONTENT}


class FakeModel:
    def encode(self, texts, **kw):
        return np.ones((len(texts), 4)) / 2.0


def test_comment_in_code_is_not_heading():
    heads = [b["text"] for b in parse_blocks(CONTENT) if b["type"] == "heading"]
    assert "# a comment, not a heading" not in heads


def _code_intact(chunks):
    return sum(CODE in c["text"] for c in chunks) == 1


def test_heading_keeps_code_whole_and_adds_section():
    chunks = heading_chunks(DOC)
    assert _code_intact(chunks)
    assert any(c["section"] == "Setup" for c in chunks)
    assert all(c["embed_text"].startswith("MQTT") for c in chunks)


def test_semantic_keeps_code_whole():
    assert _code_intact(semantic_chunks(DOC, FakeModel()))


def test_sizes_respect_budget():
    for fn in (regular_chunks, heading_chunks):
        assert max(len(c["text"]) for c in fn(DOC)) <= 1100
