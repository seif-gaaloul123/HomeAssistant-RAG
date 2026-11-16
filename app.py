import streamlit as st

from src.config import INDEXES, STRATEGIES
from src.llm import generate_answer
from src.retrieval import HybridRetriever

st.set_page_config(page_title="Home Assistant Docs RAG" , layout="wide")
st.title(" Home Assistant Documentation RAG")
st.caption("Hybrid BM25 + dense retrieval · cross-encoder reranking · local embeddings")


@st.cache_resource
def load_retriever(strategy):
    return HybridRetriever(INDEXES / strategy, use_reranker=True)


strategy = st.sidebar.selectbox("Chunking strategy", list(STRATEGIES), index=len(STRATEGIES) - 1)
mode = st.sidebar.selectbox("Retrieval", ["hybrid", "dense", "bm25"])
rerank = st.sidebar.checkbox("Cross-encoder reranking", value=True)
k = st.sidebar.slider("Retrieved chunks", 1, 10, 5)
query = st.text_input("Ask a Home Assistant documentation question")

if query:
    contexts = load_retriever(strategy).retrieve(query, mode=mode, k=k, rerank=rerank)
    key = (query, strategy, mode, rerank, k)

    st.subheader("Answer")
    # only call groq on click, otherwise every sidebar change would burn quota
    if st.button("Generate answer"):
        try:
            st.session_state["answer"] = (key, generate_answer(query, contexts))
        except RuntimeError as e:
            st.session_state["answer"] = (key, f"{e}")
        except Exception as e:
            st.session_state["answer"] = (key, f"Unexpected error: {e}")

    saved = st.session_state.get("answer")
    if saved and saved[0] == key:
        st.write(saved[1])
    else:
        st.caption("Click “Generate answer”. Retrieved sources below update live.")

    st.subheader("Retrieved sources")
    for i, c in enumerate(contexts, 1):
        label = f"{i}. {c['title']}"
        if c.get("section"):
            label += f" > {c['section']}"
        with st.expander(f"{label} — {c['source']}"):
            st.write(c["text"])
