# Medium Article RAG Assistant

A Retrieval-Augmented Generation (RAG) system over ~7,600 Medium articles, deployed as a serverless API on Vercel.

**Live URL:** `https://medium-rag-assistant-nu.vercel.app`

---

## API Endpoints

### POST `/api/prompt` — Ask a question

Send a JSON body with a `question` field. The system embeds your question, retrieves the most relevant article chunks from Pinecone, and generates an answer grounded strictly in those articles.

**curl:**
```bash
curl -X POST https://medium-rag-assistant-nu.vercel.app/api/prompt \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the best practices for writing on Medium?"}'
```

**Response:**
```json
{
  "response": "Based on the retrieved articles...",
  "context": [
    {
      "article_id": "...",
      "title": "Article Title",
      "authors": "Author Name",
      "url": "https://medium.com/...",
      "timestamp": "2019-01-01",
      "tags": "writing, productivity",
      "chunk": "The relevant passage from the article...",
      "score": 0.67
    }
  ],
  "Augmented_prompt": {
    "System": "...",
    "User": "..."
  }
}
```

**Postman:**
1. Method: `POST`
2. URL: `https://medium-rag-assistant-nu.vercel.app/api/prompt`
3. Body → raw → JSON: `{"question": "Your question here"}`

---

### GET `/api/stats` — Hyperparameter configuration

Returns the current RAG pipeline configuration.

**curl:**
```bash
curl https://medium-rag-assistant-nu.vercel.app/api/stats
```

**Response:**
```json
{
  "chunk_size": 1024,
  "overlap_ratio": 0.05,
  "top_k": 5
}
```

---

## Architecture

```
User question
    │
    ▼
Embed with text-embedding-3-small (LLMod.ai)
    │
    ▼
Query Pinecone (cosine similarity, top_k=5)
    │
    ▼
Deduplicate by article (keep highest-scoring chunk per article)
    │
    ▼
Build prompt with retrieved context
    │
    ▼
Generate answer with gpt-4o-mini (LLMod.ai)
    │
    ▼
Return JSON response
```

| Component | Value |
|---|---|
| Embedding model | `text-embedding-3-small` (1536 dims) |
| Chat model | `gpt-4o-mini` |
| Vector DB | Pinecone (cosine, serverless) |
| Index | `medium-articles` |
| Chunk size | 1024 words |
| Overlap ratio | 5% |
| Top-K retrieval | 5 chunks |

---

## Hyperparameter Tuning

The chunk size, overlap ratio, and top-K values were selected via a grid search over 45 configurations (3 chunk sizes × 4 overlap ratios × 5 top-K values) evaluated on 350 sampled articles. See [`scripts/tune_hyperparams.py`](scripts/tune_hyperparams.py) for the experiment code and [`hyperparameter_results.json`](hyperparameter_results.json) for full results.

**Scoring criteria** (per test query):
- **Diversity** — fraction of retrieved chunks from distinct articles
- **Keyword score** — presence of expected topic keywords in retrieved chunks
- **Combined score** — average of diversity and keyword score

**Winner: chunk_size=1024, overlap_ratio=0.05, top_k=5**
- `avg_combined_score`: **0.933** (highest of all 45 configurations)
- `avg_distinct_articles`: 4.75 / 5
- All 4 test query types return ≥ 3 distinct articles

The test queries cover the four capability types from the assignment spec: precise fact retrieval, multi-result topic listing, key idea summary extraction, and recommendation with justification.

---

## Deployment

Deployed on Vercel using the `@vercel/python` builder with Flask as the WSGI handler.

```json
// vercel.json
{
  "version": 2,
  "builds": [{ "src": "api/index.py", "use": "@vercel/python" }],
  "routes": [{ "src": "/(.*)", "dest": "api/index.py" }]
}
```

Required environment variables (set in Vercel project settings):
- `LLMOD_API_KEY`
- `LLMOD_BASE_URL`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
