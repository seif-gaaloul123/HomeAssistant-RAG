# HomeAssistant-RAG

A Retrieval-Augmented Generation (RAG) system over the [Home Assistant documentation](https://github.com/home-assistant/home-assistant.io), built as a **benchmark** of chunking and retrieval strategies, not just a chatbot.

It answers questions about Home Assistant integrations and core docs, cites the pages it used, and is instructed to refuse when the docs don't contain the answer.

## Highlights

- **Three chunking strategies compared head to head:** regular (fixed-size), semantic (embedding-based) and heading-aware (one chunk per section, code blocks never split), plus an ablation that removes the title/section header from the embedded text
- **Hybrid retrieval:** dense (FAISS) + sparse (BM25) fused with Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranking** of the fused candidates
- **Grounded generation** with Groq-hosted LLMs: answers must come from the retrieved chunks, cite their source, and use a fixed refusal sentence when the answer is missing
- **64-question golden set** (54 answerable, 10 unanswerable) to measure retrieval quality and test refusal behavior
- **Retrieval metrics with bootstrap confidence intervals**, so small differences are not mistaken for real ones
- **Manual review helper and RAGAS evaluation** for answer quality
- **Streamlit UI** for interactive Q&A

## How it works

```
home-assistant.io docs
 (source/_integrations, source/_docs)
            │
            ▼
   Ingest + cleaning ──► data/processed/documents.jsonl
   (strip Liquid tags / HTML / links,
    keep YAML, Jinja examples, tables, code fences)
            │
            ▼
   Chunking ──► regular / semantic / heading-aware chunks
            │
            ▼
   Indexing
   ├─ Dense:  all-MiniLM-L6-v2 embeddings → FAISS
   └─ Sparse: BM25 (rank-bm25)
            │
            ▼
   Query ─► dense top-30 + BM25 top-30 ─► RRF fusion
            │
            ▼
   Cross-encoder reranker ─► top 5 chunks
            │
            ▼
   Groq LLM (answer + [source: path])
            │
            ▼
   Streamlit UI  /  evaluation
```

## Tech stack

| Component | Tool |
|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector search | FAISS (`faiss-cpu`) |
| Keyword search | `rank-bm25` |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | Cross-encoder (ms-marco-MiniLM-L-6-v2) |
| Generation | Groq (`groq`, `langchain-groq`) |
| Evaluation | Custom retrieval metrics, RAGAS 0.2.15 |
| UI | Streamlit |
| Tests | pytest |

## Project structure

```
HomeAssistant-RAG/
├── app.py                       # Streamlit app
├── src/
│   ├── config.py                # paths, model names, chunk size
│   ├── cli.py                   # tiny command-line option reader
│   ├── cleaning.py              # markdown / Liquid cleanup
│   ├── ingest.py                # docs -> documents.jsonl (+ manifest with commit SHA)
│   ├── chunking.py              # regular, semantic and heading-aware chunking
│   ├── build_index.py           # chunks -> FAISS index per strategy
│   ├── retrieval.py             # dense, BM25, RRF, reranking
│   ├── llm.py                   # prompt, Groq call, retries, refusal check
│   ├── ask.py                   # ask one question from the terminal
│   ├── golden.py                # load and validate the golden set
│   ├── validate_golden.py       # check that golden paths exist in the corpus
│   ├── metrics.py               # Hit@K, MRR, nDCG (page level)
│   ├── evaluate_retrieval.py    # retrieval benchmark + bootstrap CIs
│   ├── generate_eval_answers.py # answers + refusal rate (resumable)
│   ├── review_answers.py        # grade answers by hand
│   └── ragas_eval.py            # RAGAS scoring
├── tests/                       # cleaning, chunking, metrics, cli
├── data/eval/                   # golden question set
├── requirements.txt
├── .env.example
└── .gitignore
```

## Getting started

Python 3.10 or 3.11 is the safest choice.

### 1. Clone and install

```bash
git clone https://github.com/seif-gaaloul123/HomeAssistant-RAG.git
cd HomeAssistant-RAG

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your Groq API key

Copy `.env.example` to `.env` and fill it in:

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=a_chat_model_available_on_your_account
```

Get a key at [console.groq.com](https://console.groq.com). Model availability differs per account (the default `llama-3.3-70b-versatile` returned 404 on mine), so list the models you can use in the Groq console and set `GROQ_MODEL` accordingly. Retrieval works without a key; only answer generation and RAGAS need it.

### 3. Get the docs

```bash
git clone --depth 1 https://github.com/home-assistant/home-assistant.io.git ../home-assistant.io
```

The pipeline uses `source/_integrations` and `source/_docs`.

### 4. Ingest and build the indexes

```bash
python -m src.ingest --repo ../home-assistant.io
python -m src.build_index --chunking all
python -m src.build_index --chunking heading --no-header   # ablation: no title/section header
```

The first run downloads the embedding and reranker models from Hugging Face. `--limit 300` on ingest is only for quick smoke tests; the sample is mostly general docs and will not contain most golden-set pages.

### 5. Ask a question or run the app

```bash
python -m src.ask --query "How do I keep passwords out of configuration.yaml?"
streamlit run app.py
```

## Evaluation

### Golden set

`data/eval/golden_questions.jsonl` contains **64 questions**, written against the docs with AI assistance. Every expected page path is checked against the corpus by `python -m src.validate_golden`.

- **54 answerable:** each mapped to the doc page that contains the answer, used to measure retrieval. About a third target near-identical pages (the many separate MQTT entity pages), where chunking and ranking choices should matter most.
- **10 unanswerable:** the docs don't cover them, used to check that the system refuses instead of hallucinating.

```json
{"question": "...", "reference": "...", "relevant_paths": ["source/_integrations/button.mqtt.markdown"]}
```
An empty `relevant_paths` marks an unanswerable question.

### Retrieval benchmark

Metrics are computed **per page**: retrieved chunks are deduplicated to pages before scoring, K = 5, 54 answerable questions. The summary reports 95% bootstrap confidence intervals. Full results are in `reports/retrieval_summary.md`; the table below shows **MRR** (rank of the first correct page, 1.0 is best).

| Retrieval | Regular | Semantic | Heading-aware |
|---|---|---|---|
| BM25 only | 0.772 | 0.759 | 0.759 |
| Dense only | 0.882 | 0.938 | 0.887 |
| Hybrid (RRF) | 0.881 | 0.907 | 0.855 |
| Dense + rerank | 0.937 | 0.910 | 0.878 |
| BM25 + rerank | 0.931 | 0.901 | 0.866 |
| Hybrid + rerank | **0.946** | 0.926 | 0.875 |

**Hit@5 saturates:** it is 1.000 for hybrid + rerank in every index and at least 0.981 for dense and hybrid, while BM25 alone reaches 0.907. That is why the comparison relies on MRR. Precision@K is capped at 0.200 (one correct page per question) and Recall@K equals Hit@K here, so neither is informative.

**Title/section header ablation** (heading-aware chunking, MRR): the header is added to the text that is embedded, keyword-indexed and reranked.

| Retrieval | With header | Without header |
|---|---|---|
| Dense only | 0.887 | 0.832 |
| Hybrid (RRF) | 0.855 | 0.849 |
| Dense + rerank | 0.878 | 0.877 |
| Hybrid + rerank | 0.875 | 0.855 |

### Answer quality

- **Refusal:** the model replied with the fixed refusal sentence on **10 of 10** unanswerable questions (exact-phrase match). Those questions are deliberately absurd, so this is a weak test; see Limitations.
- **Manual review:** <!-- fill in after running `python -m src.review_answers`, e.g. "14 of 20 answers fully correct, 4 partly supported, 2 wrong" -->
- **RAGAS (smoke test, 2 questions, Groq judge):** faithfulness 1.00, answer relevancy 0.75, context precision 1.00, context recall 1.00. Two questions say almost nothing statistically; this only confirms the pipeline runs end to end.

### Running the evaluation

```bash
python -m src.validate_golden
python -m src.evaluate_retrieval --k 5 --indexes regular semantic heading heading_nohdr
python -m src.generate_eval_answers --chunking heading --limit 20 --refusal-limit 10
python -m src.review_answers
python -m src.ragas_eval --limit 5
```

`generate_eval_answers` writes after every question and resumes if it hits a rate limit.

## Key findings

- **BM25 alone is the weakest setup** (MRR about 0.76, Hit@5 0.907). Adding dense retrieval or a reranker closes most of the gap, which fits questions that are paraphrased rather than copied from the page.
- **Best result: regular chunking with hybrid + rerank (MRR 0.946).** Semantic chunking with dense search and no reranker is close (0.938), so the reranker's value depends on the chunking.
- **Reranking improved hybrid MRR in all four indexes**, but clearly only for regular chunking (+0.065); the other gains are small (+0.006 to +0.020). For dense search the effect is mixed (up for regular, down for semantic and heading-aware).
- **Heading-aware chunking did not beat simple chunking** in this benchmark (0.875 versus 0.946 for hybrid + rerank), although the confidence intervals overlap. One untested explanation is that it produces more, smaller chunks, so several chunks of the same page compete.
- **The title/section header helps a little for dense search without a reranker** (0.887 versus 0.832) and its effect mostly disappears with the reranker (0.878 versus 0.877).

## Problems found on real data

Running the pipeline on the full docs exposed bugs that unit tests on toy input would not have caught:

- The cleaner collapsed leading spaces, **erasing YAML indentation** in configuration examples.
- Removing Liquid blocks also removed `{% raw %}` sections, **deleting the Jinja template examples**. Raw blocks are now protected before the Liquid pass.
- `{% term "configuration.yaml" %}` tags were deleted entirely, leaving sentences with the file name missing. Their text is now kept.
- The front-matter parser matched nested keys, so a `title:` inside a `related:` list **overwrote the page title**.
- Metrics were computed per chunk against page-level labels, so **recall and nDCG could exceed 1**. Results are now deduplicated to pages first.
- The first RAGAS judge returned `nan` for faithfulness because a reasoning model ran out of tokens; the judge now has an explicit `max_tokens`.

## Tests

```bash
pytest
```

Tests cover cleaning (YAML indentation, raw Jinja, tables, placeholders), chunking (code blocks stay whole), the metrics (never above 1) and the command-line helper.

## Limitations

- The golden set is small (64 questions) and was written with AI assistance; its answerable questions are paraphrased but still close to the page wording, which can favor keyword search. Treat differences smaller than the confidence intervals as ties.
- Hit@5 reaches its ceiling for the best setups, so it cannot separate them.
- Citing and refusing are **requested in the prompt**, not enforced by code: nothing checks that a cited path was actually retrieved, and nothing forces a refusal when retrieval is weak. The refusal test uses easy, obviously off-topic questions.
- RAGAS depends on the judge LLM, was run on a tiny sample because of free-tier limits, and may use the same model family as the generator.
- Latency and cost were not measured.
- Results are for one snapshot of the docs; the commit SHA is recorded in `data/processed/manifest.json`, and the indexes must be rebuilt when the docs change.

## Roadmap

- [ ] Larger, harder golden set, including realistic questions the docs do not answer
- [ ] Verify that every cited path was among the retrieved chunks
- [ ] Retrieval-score threshold for refusing
- [ ] Metadata filtering (for example by integration category) and query rewriting
- [ ] Incremental re-indexing when docs change
- [ ] Deployment (Streamlit Community Cloud / Docker)

## Author

**Seif Gaaloul**: [@seif-gaaloul123](https://github.com/seif-gaaloul123)
