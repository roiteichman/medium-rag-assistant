"""
One-time ingestion: chunk all Medium articles and upsert embeddings to Pinecone.

Hyperparameters (Phase 1 winners):
  chunk_size    = 1024 words
  overlap_ratio = 0.05

Run with --limit N first to validate, then without --limit for the full corpus.

Usage:
  conda activate medium-rag
  set LLMOD_API_KEY / LLMOD_BASE_URL / PINECONE_API_KEY / PINECONE_INDEX_NAME
  python scripts/ingest.py --limit 500    # test on first 500 articles
  python scripts/ingest.py                # full ~7,600 articles

The script is safe to re-run: Pinecone upsert is idempotent (same vector id
overwrites the previous value), so no duplicate vectors accumulate.
"""

import argparse
import csv
import os
import time
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# ---------------------------------------------------------------------------
# Hyperparameters — must match Phase 1 winners
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 1024   # words
OVERLAP_RATIO = 0.05

# ---------------------------------------------------------------------------
# Operational constants
# ---------------------------------------------------------------------------
EMBED_BATCH   = 256    # texts per LangChain embedding API call (batch_size)
UPSERT_BATCH  = 100    # vectors per Pinecone upsert call
CHUNK_BATCH   = 2000   # chunks embedded + upserted together (progress granularity)

DIMENSION     = 1536   # text-embedding-3-small
METRIC        = "cosine"

CSV_PATH        = Path(__file__).parent.parent.parent / "Dataset" / "medium-english-50mb.csv"
EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
LLMOD_BASE      = os.environ.get("LLMOD_BASE_URL", "https://api.llmod.ai/v1")
COST_PER_1M     = 0.02  # USD per 1M tokens


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def make_embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        openai_api_key=os.environ["LLMOD_API_KEY"],
        openai_api_base=LLMOD_BASE,
        model=EMBEDDING_MODEL,
        chunk_size=EMBED_BATCH,
    )


def get_or_create_index(pc: Pinecone, name: str):
    existing = [idx.name for idx in pc.list_indexes()]
    if name not in existing:
        print(f"Creating Pinecone index '{name}' (dim=1536, cosine, serverless aws/us-east-1) ...")
        pc.create_index(
            name=name,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("  Waiting for index to be ready ...")
        for _ in range(30):
            desc = pc.describe_index(name)
            ready = getattr(desc.status, "ready", None) or (
                isinstance(desc.status, dict) and desc.status.get("ready")
            )
            if ready:
                break
            time.sleep(2)
        print("  Index ready.")
    else:
        print(f"Using existing Pinecone index '{name}'.")
    return pc.Index(name)


# ---------------------------------------------------------------------------
# Chunking — word-based, matches tune_hyperparams.py exactly
# ---------------------------------------------------------------------------

def chunk_text(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, int(CHUNK_SIZE * (1 - OVERLAP_RATIO)))
    chunks, start = [], 0
    while start < len(words):
        chunk = " ".join(words[start : start + CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk)
        start += step
        if start + CHUNK_SIZE > len(words) and start < len(words):
            last = " ".join(words[start:])
            if last.strip():
                chunks.append(last)
            break
    return chunks


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_articles(limit=None) -> list[dict]:
    print(f"Loading articles from {CSV_PATH} ...")
    articles = []
    with open(CSV_PATH, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            if not row.get("text", "").strip():
                continue
            articles.append({
                "article_id": str(i),
                "title":      row.get("title",     "").strip(),
                "text":       row.get("text",      "").strip(),
                "authors":    row.get("authors",   "").strip(),
                "url":        row.get("url",       "").strip(),
                "timestamp":  row.get("timestamp", "").strip(),
                "tags":       row.get("tags",      "").strip(),
            })
    print(f"  Loaded {len(articles)} articles.")
    return articles


def build_records(articles: list[dict]) -> list[tuple]:
    """Return list of (vector_id, chunk_text, metadata) for all chunks."""
    records = []
    for art in articles:
        text = f"{art['title']} {art['text']}"
        for ci, chunk in enumerate(chunk_text(text)):
            records.append((
                f"{art['article_id']}_chunk_{ci}",
                chunk,
                {
                    "article_id":  art["article_id"],
                    "title":       art["title"],
                    "authors":     art["authors"],
                    "url":         art["url"],
                    "timestamp":   art["timestamp"],
                    "tags":        art["tags"],
                    "chunk_index": ci,
                    "chunk":       chunk,  # stored inline — no separate doc store needed
                },
            ))
    return records


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def ingest(limit=None):
    embedder = make_embedder()

    pc    = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    iname = os.environ.get("PINECONE_INDEX_NAME", "medium-articles")
    index = get_or_create_index(pc, iname)

    articles = load_articles(limit=limit)
    records  = build_records(articles)
    total    = len(records)

    total_words = sum(len(r[1].split()) for r in records)
    est_tokens  = total_words * 1.33
    print(f"\n{total} chunks from {len(articles)} articles")
    print(f"  ~{est_tokens:,.0f} estimated tokens  ~${est_tokens / 1_000_000 * COST_PER_1M:.3f} USD")

    # Process in CHUNK_BATCH-sized groups: embed then immediately upsert
    upserted = 0
    for batch_start in range(0, total, CHUNK_BATCH):
        batch = records[batch_start : batch_start + CHUNK_BATCH]
        texts = [r[1] for r in batch]

        end = min(batch_start + CHUNK_BATCH, total)
        print(f"\n[{batch_start+1}–{end} / {total}] Embedding {len(texts)} chunks ...")
        vectors = embedder.embed_documents(texts)

        # Upsert this batch in UPSERT_BATCH-sized sub-batches
        for j in range(0, len(batch), UPSERT_BATCH):
            sub      = batch[j : j + UPSERT_BATCH]
            sub_vecs = vectors[j : j + UPSERT_BATCH]
            index.upsert(vectors=[
                {"id": vid, "values": vec, "metadata": meta}
                for (vid, _, meta), vec in zip(sub, sub_vecs)
            ])
            upserted += len(sub)

        pct = upserted / total * 100
        print(f"  {upserted}/{total} vectors upserted ({pct:.1f}%)")

    stats = index.describe_index_stats()
    print(f"\nIngestion complete.")
    print(f"  Vectors upserted this run : {upserted}")
    print(f"  Total vectors in index    : {stats['total_vector_count']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest Medium articles into Pinecone.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Ingest only first N articles (omit for full ~7,600)")
    args = ap.parse_args()
    ingest(limit=args.limit)
