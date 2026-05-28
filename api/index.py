import os

from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from pinecone import Pinecone

# ---------------------------------------------------------------------------
# Vercel: allow up to 60 s (embed + query + generate exceeds 10 s default)
# ---------------------------------------------------------------------------
max_duration = 60

# ---------------------------------------------------------------------------
# Hyperparameters — must match ingest.py and /api/stats response
# ---------------------------------------------------------------------------
TOP_K           = 5
EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
CHAT_MODEL      = "4UHRUIN-gpt-5-mini"

SYSTEM_PROMPT = (
    "You are a Medium-article assistant that answers questions strictly and only "
    "based on the Medium articles dataset context provided to you (metadata and "
    "article passages). You must not use any external knowledge, the open internet, "
    "or information that is not explicitly contained in the retrieved context. "
    "If the answer cannot be determined from the provided context, respond: "
    "'I don't know based on the provided Medium articles data.' "
    "Always explain your answer using the given context, quoting or paraphrasing "
    "the relevant article passage or metadata when helpful. "
    "Response style: Be concise and direct. For multi-article requests, present "
    "results as a numbered list. Always attribute information to a specific article title."
)

# ---------------------------------------------------------------------------
# Lazy-initialised singletons (created once per warm Lambda container)
# ---------------------------------------------------------------------------
_embedder = None
_llm      = None
_index    = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbeddings(
            openai_api_key=os.environ["LLMOD_API_KEY"],
            openai_api_base=os.environ.get("LLMOD_BASE_URL", "https://api.llmod.ai/v1"),
            model=EMBEDDING_MODEL,
            chunk_size=256,
        )
    return _embedder


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            openai_api_key=os.environ["LLMOD_API_KEY"],
            openai_api_base=os.environ.get("LLMOD_BASE_URL", "https://api.llmod.ai/v1"),
            model=CHAT_MODEL,
        )
    return _llm


def _get_index():
    global _index
    if _index is None:
        pc     = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        iname  = os.environ.get("PINECONE_INDEX_NAME", "medium-articles")
        _index = pc.Index(iname)
    return _index


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------

def _embed_query(question: str) -> list[float]:
    return _get_embedder().embed_query(question)


def _retrieve(query_vec: list[float]) -> list[dict]:
    results = _get_index().query(
        vector=query_vec,
        top_k=TOP_K,
        include_metadata=True,
    )
    chunks = []
    for match in results.matches:
        meta = match.metadata or {}
        chunks.append({
            "article_id": meta.get("article_id", match.id),
            "title":      meta.get("title", ""),
            "authors":    meta.get("authors", ""),
            "url":        meta.get("url", ""),
            "timestamp":  meta.get("timestamp", ""),
            "tags":       meta.get("tags", ""),
            "chunk":      meta.get("chunk", ""),
            "score":      match.score,
        })
    return chunks


def _dedup_by_article(chunks: list[dict]) -> list[dict]:
    """Keep the highest-scoring chunk per article_id."""
    seen: dict[str, dict] = {}
    for c in chunks:
        aid = c["article_id"]
        if aid not in seen or c["score"] > seen[aid]["score"]:
            seen[aid] = c
    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)


def _build_user_prompt(question: str, context: list[dict]) -> str:
    lines = ["Context from retrieved Medium articles:", "---"]
    for i, c in enumerate(context, 1):
        lines.append(
            f"[CHUNK {i}] Article: \"{c['title']}\" (ID: {c['article_id']}) | "
            f"Authors: {c['authors']} | Published: {c['timestamp']} | "
            f"Tags: {c['tags']} | URL: {c['url']}"
        )
        lines.append(c["chunk"])
        lines.append("")
    lines.append("---")
    lines.append(f"Question: {question}")
    return "\n".join(lines)


def _generate(system: str, user: str) -> str:
    response = _get_llm().invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    return response.content


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "ok",
        "endpoints": {
            "POST /api/prompt": "RAG query — body: {\"question\": \"...\"}",
            "GET  /api/stats":  "Hyperparameter configuration",
        }
    }), 200


@app.route("/api/prompt", methods=["POST"])
@app.route("/prompt", methods=["POST"])
def prompt():
    body = request.get_json(force=True, silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'question' field"}), 400

    try:
        query_vec = _embed_query(question)
        raw       = _retrieve(query_vec)
        context   = _dedup_by_article(raw)
        user_msg  = _build_user_prompt(question, context)
        answer    = _generate(SYSTEM_PROMPT, user_msg)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "response": answer,
        "context": [
            {
                "article_id": c["article_id"],
                "title":      c["title"],
                "chunk":      c["chunk"],
                "score":      c["score"],
                "authors":    c["authors"],
                "url":        c["url"],
                "timestamp":  c["timestamp"],
                "tags":       c["tags"],
            }
            for c in context
        ],
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User":   user_msg,
        },
    }), 200


@app.route("/api/stats", methods=["GET"])
@app.route("/stats", methods=["GET"])
def stats():
    return jsonify({
        "chunk_size":    1024,
        "overlap_ratio": 0.05,
        "top_k":         5,
    }), 200
