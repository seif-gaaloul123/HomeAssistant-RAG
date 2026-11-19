from src.cleaning import clean_markdown

DOC = """---
title: Templating
description: Jinja examples
---
{% raw %}
```yaml
sensor:
  - platform: template
    value: "{{ states('sensor.x') }}"
```
{% endraw %}
{% configuration %}
host:
  description: The host <your_ip>
{% endconfiguration %}

| a | b |
|---|---|
| 1 | 2 |

See [the docs](https://example.com) for more.
"""


def test_cleaning_keeps_structure():
    meta, body = clean_markdown(DOC)
    assert body.startswith("# Templating")
    assert "  - platform: template" in body          # YAML indentation survives
    assert "{{ states('sensor.x') }}" in body        # raw Jinja survives
    assert "  description: The host <your_ip>" in body  # placeholders and nesting survive
    assert "|---|---|" in body                       # table separator survives
    assert "See the docs for more." in body          # links become text
    assert "{%" not in body


def test_term_tag_and_nested_title():
    raw = """---
title: MQTT alarm
related:
  - docs: /docs/x/
    title: Configuration file
---
Add the following to your {% term "configuration.yaml" %} file, see {% term YAML %}.
"""
    meta, body = clean_markdown(raw)
    assert meta["title"] == "MQTT alarm"
    assert "your configuration.yaml file" in body
    assert "see YAML." in body
