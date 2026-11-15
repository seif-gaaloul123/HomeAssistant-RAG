import os
import time

from groq import APIConnectionError, APIStatusError, Groq, RateLimitError

from .config import GROQ_MODEL

REFUSAL = "The documentation context does not contain this information."


def format_contexts(contexts):
    blocks = []
    for c in contexts:
        label = f"{c['source']} | {c['title']}"
        if c.get("section"):
            label += f" > {c['section']}"
        blocks.append(f"[Source: {label}]\n{c['text']}")
    return "\n\n".join(blocks)


def is_refusal(answer):
    return REFUSAL.lower() in answer.lower()


def generate_answer(question, contexts, retries=5):
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Retrieval works without it.")
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    prompt = f'''Answer the question using ONLY the supplied Home Assistant documentation.
If the context does not contain the answer, reply exactly: "{REFUSAL}"
Do not invent configuration keys, services, entities or steps.
Cite sources inline using [source: path].

QUESTION:
{question}

DOCUMENTATION CONTEXT:
{format_contexts(contexts)}'''

    messages = [
        {"role": "system", "content": "You are a careful Home Assistant documentation assistant."},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(model=GROQ_MODEL, temperature=0.1, messages=messages)
            return resp.choices[0].message.content
        except (RateLimitError, APIConnectionError) as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Groq request failed after {retries} attempts: {e}") from e
            time.sleep(2 ** attempt * 2)
        except APIStatusError as e:
            raise RuntimeError(f"Groq API error: {e}") from e
