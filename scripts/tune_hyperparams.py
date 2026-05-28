"""
Hyperparameter tuning experiment for the Medium Article RAG pipeline.

Runs a grid search over chunk_size × overlap × top_k on a 200-article sample.
Grid values are informed by Lecture 3 slides 11-13 best practices for long articles.
Uses local numpy cosine similarity (NO Pinecone, NO chat model) to keep cost near zero.
Embeddings are cached to disk so re-runs with the same chunk config are free.

Uses LangChain's OpenAIEmbeddings pointed at LLMod.ai — consistent with production code.

Usage (from medium-rag-assistant/ directory):
    set LLMOD_API_KEY=<your key>
    python scripts/tune_hyperparams.py

Results are printed as a ranked table and saved to hyperparameter_results.json.
"""

import csv
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
from langchain_openai import OpenAIEmbeddings

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_SIZE   = 200
CSV_PATH      = Path(__file__).parent.parent.parent / "Dataset" / "medium-english-50mb.csv"
CACHE_DIR     = Path("experiment_cache")
RESULTS_PATH  = Path("hyperparameter_results.json")
LLMOD_BASE    = os.environ.get("LLMOD_BASE_URL", "https://api.llmod.ai/v1")

EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"

# Grid to search — values from Lecture 3 slides 11-13:
#   chunk_size 512–1024 for long articles (Medium posts avg ~600-800 words;
#     2048 would collapse most articles to a single chunk, killing passage retrieval)
#   overlap 5–15% for long articles ("less expensive")
#   top_k 8–12 for research/long text; 5 added as lower-bound comparison
CHUNK_SIZES = [512, 768, 1024]  # words used as token proxy
OVERLAPS    = [0.05, 0.10, 0.15]
TOP_KS      = [5, 8, 10, 12]

# Test queries: (name, query_text, expected_keywords)
# One per assignment question type; keywords are a retrieval-quality proxy.
TEST_QUERIES = [
    (
        "precise_fact",
        "Find an article about using Python for data analysis. Give the title and author.",
        ["python", "data", "analysis", "pandas", "numpy", "dataframe"],
    ),
    (
        "multi_result",
        "List 3 articles about machine learning or artificial intelligence.",
        ["machine learning", "deep learning", "neural", "ai", "model", "algorithm"],
    ),
    (
        "summary",
        "Find an article discussing remote work or working from home and summarise it.",
        ["remote", "work", "home", "office", "team", "distributed"],
    ),
    (
        "recommendation",
        "I want practical advice on building productive daily habits. Recommend an article.",
        ["habit", "productivity", "routine", "morning", "goal", "focus"],
    ),
]

# ---------------------------------------------------------------------------
# LangChain embeddings client
# ---------------------------------------------------------------------------

def make_embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        openai_api_key=os.environ["LLMOD_API_KEY"],
        openai_api_base=LLMOD_BASE,
        model=EMBEDDING_MODEL,
        # chunk_size = batch size (Lecture 4, slide 27: batch_size=256).
        # LangChain splits the full list into batches of 256 internally.
        # NEVER call embed_query() in a loop over chunks — always pass the
        # full list to embed_documents() and let LangChain batch it.
        chunk_size=256,
    )


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size_words: int, overlap: float) -> list[str]:
    """
    Split text into overlapping chunks by word count.
    chunk_size_words ≈ tokens (1 token ≈ 0.75 words, so 256 words ≈ 340 tokens —
    well within the 1024-token assignment limit at any grid point).
    """
    words = text.split()
    if not words:
        return []
    step = max(1, int(chunk_size_words * (1 - overlap)))
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size_words])
        if chunk.strip():
            chunks.append(chunk)
        start += step
        if start + chunk_size_words > len(words) and start < len(words):
            # last partial chunk
            chunk = " ".join(words[start:])
            if chunk.strip():
                chunks.append(chunk)
            break
    return chunks


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_topk(query_vec: np.ndarray, corpus: np.ndarray, k: int) -> list[int]:
    norms = np.linalg.norm(corpus, axis=1, keepdims=True).clip(min=1e-9)
    normed = corpus / norms
    q = query_vec / max(np.linalg.norm(query_vec), 1e-9)
    scores = normed @ q
    return np.argsort(scores)[::-1][:k].tolist()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def diversity_score(top_indices: list[int], chunk_to_article: list[int]) -> float:
    articles = [chunk_to_article[i] for i in top_indices]
    return len(set(articles)) / len(articles)


def keyword_score(top_indices: list[int], chunks: list[str], keywords: list[str], n: int = 3) -> float:
    hits = sum(
        1 for idx in top_indices[:n]
        if any(kw.lower() in chunks[idx].lower() for kw in keywords)
    )
    return hits / min(n, len(top_indices))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sample() -> list[dict]:
    print(f"Loading {SAMPLE_SIZE} articles from {CSV_PATH} ...")
    with open(CSV_PATH, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        all_rows = [row for row in reader if row.get("text", "").strip()]

    random.seed(42)
    sample = random.sample(all_rows, min(SAMPLE_SIZE, len(all_rows)))
    articles = [
        {
            "article_id": str(i),
            "title":   row.get("title", "").strip(),
            "text":    row.get("text",  "").strip(),
            "authors": row.get("authors", "").strip(),
        }
        for i, row in enumerate(sample)
    ]
    print(f"  Loaded {len(articles)} articles.")
    return articles


# ---------------------------------------------------------------------------
# Per-config corpus (with disk cache to avoid re-spending money)
# ---------------------------------------------------------------------------

def build_corpus(
    articles: list[dict],
    chunk_size: int,
    overlap: float,
    embedder: OpenAIEmbeddings,
) -> tuple[list[str], list[int], np.ndarray]:
    key        = f"cs{chunk_size}_ov{int(overlap * 100):02d}"
    cache_file = CACHE_DIR / f"{key}.pkl"
    CACHE_DIR.mkdir(exist_ok=True)

    if cache_file.exists():
        print(f"  [cache hit] {key}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"  Chunking: chunk_size={chunk_size} words, overlap={overlap} ...")
    chunks: list[str] = []
    chunk_to_article: list[int] = []
    for i, article in enumerate(articles):
        text = f"{article['title']} {article['text']}"
        for ch in chunk_text(text, chunk_size, overlap):
            chunks.append(ch)
            chunk_to_article.append(i)

    print(f"  {len(chunks)} chunks total. Embedding via LangChain ...")
    # embed_documents() handles batching internally (chunk_size=256 set above)
    vectors = embedder.embed_documents(chunks)
    embeddings = np.array(vectors, dtype=np.float32)

    with open(cache_file, "wb") as f:
        pickle.dump((chunks, chunk_to_article, embeddings), f)
    print(f"  Saved → {cache_file}")
    return chunks, chunk_to_article, embeddings


# ---------------------------------------------------------------------------
# Query vector cache
# ---------------------------------------------------------------------------

def get_query_vectors(embedder: OpenAIEmbeddings) -> dict[str, np.ndarray]:
    cache_file = CACHE_DIR / "query_vecs.pkl"
    CACHE_DIR.mkdir(exist_ok=True)

    if cache_file.exists():
        print("[cache hit] query vectors")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("Embedding test queries ...")
    # Only 4 queries — embed_documents() batches them in one API call.
    q_texts = [q_text for _, q_text, _ in TEST_QUERIES]
    vecs    = embedder.embed_documents(q_texts)
    result  = {
        q_name: np.array(vec, dtype=np.float32)
        for (q_name, _, _), vec in zip(TEST_QUERIES, vecs)
    }
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    return result


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run():
    embedder   = make_embedder()
    articles   = load_sample()
    query_vecs = get_query_vectors(embedder)

    results = []
    total   = len(CHUNK_SIZES) * len(OVERLAPS) * len(TOP_KS)
    done    = 0

    for chunk_size in CHUNK_SIZES:
        for overlap in OVERLAPS:
            chunks, chunk_to_article, embeddings = build_corpus(
                articles, chunk_size, overlap, embedder
            )

            for top_k in TOP_KS:
                row: dict = {
                    "chunk_size":    chunk_size,
                    "overlap_ratio": overlap,
                    "top_k":         top_k,
                    "queries":       {},
                }

                for q_name, _q_text, q_keywords in TEST_QUERIES:
                    top_idx = cosine_topk(query_vecs[q_name], embeddings, top_k)
                    div = diversity_score(top_idx, chunk_to_article)
                    kw  = keyword_score(top_idx, chunks, q_keywords)
                    row["queries"][q_name] = {
                        "diversity":     round(div, 3),
                        "keyword_score": round(kw,  3),
                        "combined":      round((div + kw) / 2, 3),
                    }

                avg = sum(v["combined"] for v in row["queries"].values()) / len(TEST_QUERIES)
                row["avg_combined_score"] = round(avg, 3)
                results.append(row)

                done += 1
                print(f"  [{done}/{total}] cs={chunk_size} ov={overlap} k={top_k}  avg={avg:.3f}")

    results.sort(key=lambda r: r["avg_combined_score"], reverse=True)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {RESULTS_PATH}")

    # ---- Summary table ----
    print(f"\n{'chunk_size':>11} {'overlap':>8} {'top_k':>6}  "
          f"{'diversity':>10} {'kw_score':>9} {'combined':>9}")
    print("─" * 60)
    for r in results:
        avg_div = sum(v["diversity"]     for v in r["queries"].values()) / len(TEST_QUERIES)
        avg_kw  = sum(v["keyword_score"] for v in r["queries"].values()) / len(TEST_QUERIES)
        marker  = " ◄ BEST" if r is results[0] else ""
        print(f"{r['chunk_size']:>11} {r['overlap_ratio']:>8.1f} {r['top_k']:>6}  "
              f"{avg_div:>10.3f} {avg_kw:>9.3f} {r['avg_combined_score']:>9.3f}{marker}")

    best = results[0]
    print(f"\nRecommended config:")
    print(f"  chunk_size    = {best['chunk_size']}")
    print(f"  overlap_ratio = {best['overlap_ratio']}")
    print(f"  top_k         = {best['top_k']}")
    print(f"\nUpdate plan.md hyperparameter table and api/stats.py with these values.")


if __name__ == "__main__":
    run()
