"""
Hyperparameter tuning experiment for the Medium Article RAG pipeline.

Runs a grid search over chunk_size × overlap × top_k on a sample of articles.
Grid values are informed by Lecture 3 slides 11-13 best practices for long articles.
Uses local numpy cosine similarity (NO Pinecone, NO chat model) to keep cost near zero.
Embeddings are cached to disk so re-runs with the same chunk config are free.

Cache keys include the sample size so changing SAMPLE_SIZE safely invalidates old caches.

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

SAMPLE_SIZE   = 350
CSV_PATH      = Path(__file__).parent.parent.parent / "Dataset" / "medium-english-50mb.csv"
CACHE_DIR     = Path("experiment_cache")
RESULTS_PATH  = Path("hyperparameter_results.json")
LLMOD_BASE    = os.environ.get("LLMOD_BASE_URL", "https://api.llmod.ai/v1")

EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"

# Grid to search — values from Lecture 3 slides 11-13 + expanded to cover the
# 0.20 overlap region that published implementations commonly use:
#   chunk_size 512-1024 for long articles
#   overlap 5-20% (spec allows up to 30%)
#   top_k 5-15; assignment allows up to 30
CHUNK_SIZES = [512, 768, 1024]
OVERLAPS    = [0.05, 0.10, 0.15, 0.20]
TOP_KS      = [5, 8, 10, 12, 15]

# Test queries — exact examples from the assignment spec (query capability section):
#   1. Precise fact retrieval
#   2. Multi-result topic listing (must return 3 distinct articles)
#   3. Key idea summary extraction
#   4. Recommendation with evidence-based justification
TEST_QUERIES = [
    (
        "precise_fact",
        "Find an article that reframes marketing as a conversation with readers, "
        "aimed at writers who find self-promotion uncomfortable. Provide the title and author.",
        ["marketing", "conversation", "writers", "self-promotion", "uncomfortable", "readers"],
    ),
    (
        "multi_result",
        "List exactly 3 articles about education. Return only the titles.",
        ["education", "learning", "school", "student", "teacher", "knowledge", "curriculum"],
    ),
    (
        "summary",
        "Find an article that argues past pandemics (such as the bubonic plague) can spur "
        "innovation and recovery, and summarise its central argument.",
        ["pandemic", "plague", "innovation", "recovery", "history", "disease", "bubonic"],
    ),
    (
        "recommendation",
        "I want practical, beginner-friendly advice on building habits that actually stick. "
        "Which article would you recommend, and why?",
        ["habit", "practical", "beginner", "stick", "productivity", "routine", "advice"],
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
    Word count is used as an approximate token proxy (1 token ~= 0.75 words).
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
    """Fraction of top-k chunks that come from distinct articles."""
    articles = [chunk_to_article[i] for i in top_indices]
    return len(set(articles)) / len(articles)


def distinct_articles(top_indices: list[int], chunk_to_article: list[int]) -> int:
    """Absolute count of distinct articles in top-k (critical for multi-result queries)."""
    return len({chunk_to_article[i] for i in top_indices})


def keyword_score(top_indices: list[int], chunks: list[str], keywords: list[str], n: int = 3) -> float:
    """Fraction of top-n chunks containing at least one expected keyword."""
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
# Cache key includes sample size so changing SAMPLE_SIZE invalidates old caches.
# ---------------------------------------------------------------------------

def build_corpus(
    articles: list[dict],
    chunk_size: int,
    overlap: float,
    embedder: OpenAIEmbeddings,
) -> tuple[list[str], list[int], np.ndarray]:
    key        = f"cs{chunk_size}_ov{int(overlap * 100):02d}_n{SAMPLE_SIZE}"
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
    vectors = embedder.embed_documents(chunks)
    embeddings = np.array(vectors, dtype=np.float32)

    with open(cache_file, "wb") as f:
        pickle.dump((chunks, chunk_to_article, embeddings), f)
    print(f"  Saved -> {cache_file}")
    return chunks, chunk_to_article, embeddings


# ---------------------------------------------------------------------------
# Query vector cache (also keyed by sample size to stay consistent)
# ---------------------------------------------------------------------------

def get_query_vectors(embedder: OpenAIEmbeddings) -> dict[str, np.ndarray]:
    cache_file = CACHE_DIR / f"query_vecs_n{SAMPLE_SIZE}.pkl"
    CACHE_DIR.mkdir(exist_ok=True)

    if cache_file.exists():
        print("[cache hit] query vectors")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("Embedding test queries ...")
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
                    div     = diversity_score(top_idx, chunk_to_article)
                    n_art   = distinct_articles(top_idx, chunk_to_article)
                    kw      = keyword_score(top_idx, chunks, q_keywords)
                    row["queries"][q_name] = {
                        "diversity":          round(div,   3),
                        "distinct_articles":  n_art,
                        "keyword_score":      round(kw,    3),
                        "combined":           round((div + kw) / 2, 3),
                    }

                avg_combined  = sum(v["combined"]          for v in row["queries"].values()) / len(TEST_QUERIES)
                avg_n_art     = sum(v["distinct_articles"] for v in row["queries"].values()) / len(TEST_QUERIES)
                # Multi-result queries require >= 3 distinct articles; flag configs that meet it for all 4 queries
                meets_3art    = all(v["distinct_articles"] >= 3 for v in row["queries"].values())

                row["avg_combined_score"]  = round(avg_combined, 3)
                row["avg_distinct_articles"] = round(avg_n_art, 2)
                row["meets_3article_floor"] = meets_3art
                results.append(row)

                done += 1
                flag = " [OK]" if meets_3art else " [<3art]"
                print(f"  [{done:02d}/{total}] cs={chunk_size} ov={overlap:.2f} k={top_k:2d}  "
                      f"avg={avg_combined:.3f}  n_art={avg_n_art:.1f}{flag}")

    results.sort(key=lambda r: (r["meets_3article_floor"], r["avg_combined_score"]), reverse=True)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {RESULTS_PATH}")

    # ---- Summary table ----
    header = f"{'cs':>6} {'ov':>5} {'k':>4}  {'diversity':>9} {'kw_score':>8} {'n_art':>6} {'combined':>9}  {'3-art?':>6}"
    print(f"\n{header}")
    print("-" * len(header))
    for r in results[:20]:
        avg_div = sum(v["diversity"]     for v in r["queries"].values()) / len(TEST_QUERIES)
        avg_kw  = sum(v["keyword_score"] for v in r["queries"].values()) / len(TEST_QUERIES)
        ok      = "YES" if r["meets_3article_floor"] else "no"
        marker  = " << BEST" if r is results[0] else ""
        print(f"{r['chunk_size']:>6} {r['overlap_ratio']:>5.2f} {r['top_k']:>4}  "
              f"{avg_div:>9.3f} {avg_kw:>8.3f} {r['avg_distinct_articles']:>6.1f} "
              f"{r['avg_combined_score']:>9.3f}  {ok:>6}{marker}")

    best = results[0]
    print(f"\nRecommended config:")
    print(f"  chunk_size    = {best['chunk_size']}")
    print(f"  overlap_ratio = {best['overlap_ratio']}")
    print(f"  top_k         = {best['top_k']}")
    print(f"  avg_combined  = {best['avg_combined_score']}")
    print(f"  avg_distinct_articles = {best['avg_distinct_articles']}")
    print(f"\nUpdate plan.md hyperparameter table and api/stats.py with these values.")


if __name__ == "__main__":
    run()
