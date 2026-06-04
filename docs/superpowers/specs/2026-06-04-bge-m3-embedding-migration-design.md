# BGE-M3 Embedding Migration

**Date:** 2026-06-04
**Status:** Approved
**Branch:** `feature/bge-m3-embeddings`

## Goal

Replace the Gemini embedding model (`gemini-embedding-2`, 768-dim) with **BGE-M3** (1024-dim), served as a FastAPI endpoint on a remote SSH server. The pipeline must continue to produce the same candidate-ranking artifacts (BM25 index, FAISS index, embeddings, query embedding, feature scores, rerank scores) and the same `submission.csv` output — only the source of dense embeddings changes.

## Context

- The precompute phase currently calls Google's `gemini-embedding-2` via `google-genai`. Quotas and rate limits on the free tier are restrictive (90 req/min) and contribute a large share of precompute wall-clock time.
- BGE-M3 is already running on a remote server (`192.168.31.246:5000`) exposing `POST /embed`. The server returns 1024-dim vectors.
- The ranking phase is offline and must not need network access. Embedding artifacts (`embeddings.npy`, `query_embedding.npy`, `faiss.index`) are precomputed once and read from disk.
- The DeepSeek re-ranker is unaffected by this change.

## Approach

Direct HTTP call to the BGE-M3 endpoint, replacing the Gemini client internals while preserving the function signature, batching, checkpointing, and retry behavior.

Trade-off: this is not an abstract embedding-client interface, so swapping models in the future still requires code edits. Simpler for now; can abstract later.

## File-Level Changes

### 1. `src/config.py`

**Add:**
```python
BGE_M3_API_URL = "http://192.168.31.246:5000/embed"
BGE_M3_EMBEDDING_DIM = 1024
BGE_M3_REQUEST_TIMEOUT = 60  # seconds
BGE_M3_MAX_RETRIES = 5
```

**Modify:**
```python
EMBEDDING_MODEL = "bge-m3"          # was "gemini-embedding-2"
EMBEDDING_BATCH_SIZE = 32           # was 100 — smaller batches reduce server memory pressure
EMBEDDING_CHECKPOINT_EVERY = 10
```

Keep `EMBEDDING_BATCH_SIZE` name and checkpointing constants so downstream code that imports them does not break.

### 2. `src/retrieval.py`

**Remove:** `from google import genai` (line 13) and all references to `google.genai.types`.

**Add:** `import requests` and `from src.config import BGE_M3_API_URL, BGE_M3_EMBEDDING_DIM, BGE_M3_REQUEST_TIMEOUT, BGE_M3_MAX_RETRIES`.

**Replace `compute_gemini_embeddings()` (lines 35-163) with `compute_bge_m3_embeddings()`:**

Function signature unchanged: `compute_bge_m3_embeddings(texts: list[str], checkpoint_path: str = None) -> np.ndarray`.

Behavior:
- Drop the `api_key` parameter (not needed for BGE-M3 endpoint).
- Checkpoint loading/saving logic preserved, but use `BGE_M3_EMBEDDING_DIM` (1024) instead of hard-coded 768.
- Each batch: call `requests.post(BGE_M3_API_URL, json={"inputs": batch_texts}, timeout=BGE_M3_REQUEST_TIMEOUT)`. Expect response `[[...], [...]]` — a JSON array of vectors (Hugging Face TEI / sentence-transformers default).
- Concurrency: keep `ThreadPoolExecutor(max_workers=5)`. The remote server is local-network fast, so the existing concurrency model still works.
- Drop the Gemini-style rate limiter (Gemini free tier needed 67s/100 docs; BGE-M3 on local network is not throttled).
- Retry on `requests.exceptions.RequestException` or HTTP status >= 500, with exponential backoff `min(2 ** retries, 60)`. Drop the special-case 30s wait for 429.
- On final failure, return zero vectors of dim 1024 for that batch.
- Progress bar text: `"Embedding (BGE-M3)"`.

`build_faiss_index()` and `hybrid_retrieve()` already accept the embedding dim from the array shape — no changes needed.

### 3. `src/pipeline.py`

**In `run_precompute()`:**
- Replace the import on line 49:
  ```python
  from src.retrieval import build_bm25_index_save, compute_bge_m3_embeddings, build_faiss_index_save
  ```
- Replace lines 59-61:
  ```python
  embeddings = compute_bge_m3_embeddings(
      corpus, checkpoint_path=embedding_checkpoint
  )
  ```
- Replace the query-embedding block (lines 64-79). The Gemini client is no longer used; instead:
  ```python
  import requests
  from src.config import BGE_M3_API_URL, BGE_M3_REQUEST_TIMEOUT
  resp = requests.post(
      BGE_M3_API_URL,
      json={"inputs": [ideal_profile]},
      timeout=BGE_M3_REQUEST_TIMEOUT,
  )
  resp.raise_for_status()
  query_embedding = np.array(resp.json()[0], dtype=np.float32)
  np.save(os.path.join(output_dir, "query_embedding.npy"), query_embedding)
  ```
- Remove `from google import genai` and `from google.genai import types` (lines 64-65) and the `genai` import elsewhere if no longer used.
- Remove `EMBEDDING_MODEL` from the `from src.config` import on line 13.

**In `run_ranking()`:** No changes — reads `embeddings.npy` and `query_embedding.npy` from disk, dim-agnostic.

### 4. `precompute.py`

- Drop the `google_api_key` argument and the `google_api_key or os.environ.get("GOOGLE_API_KEY")` lookup (lines 17, 22, 25-28).
- Drop the `--google-api-key` and `--api-key` argparse arguments.
- Drop the `google_api_key` parameter from the `run_precompute()` call.
- Keep the DeepSeek API key gate as-is.
- No network is required for embedding-related steps; the BGE-M3 endpoint is on the local network and considered part of the precompute environment.

### 5. `.env`

**Remove:** `GOOGLE_API_KEY=...` (line 1).

**Keep:** `DEEPSEEK_API_KEY=...` (line 2).

**Add (optional, for clarity):**
```
BGE_M3_API_URL=http://192.168.31.246:5000/embed
```

If the URL lives in `config.py` as a constant, the `.env` line is unnecessary. Decision: keep the URL in `config.py` as a constant; do not add it to `.env`. The server URL is a deployment detail, not a secret.

### 6. `sample/requirements.txt`

**Remove:** `google-genai>=0.7.0` (line 1).

**Add:** `requests>=2.31` (line 2 or wherever alphabetically appropriate).

All other dependencies unchanged.

## What Stays The Same

- BM25 index construction and serialization (`build_bm25_index_save`) — untouched.
- FAISS index build/search — untouched, dim detected from input array.
- Hybrid retrieval (RRF fusion, title safety net) — untouched.
- Feature scoring (`scoring.py`), honeypot detection (`honeypot.py`), reasoning (`reasoning.py`) — untouched.
- DeepSeek re-ranking (`reranking.py`) — untouched.
- `rank.py`, `validate.py`, `app.py` — untouched.
- `submission.csv` schema — untouched.

## Error Handling

- Network errors to BGE-M3 endpoint: 5 retries with exponential backoff, then zero-vector fallback for the batch.
- HTTP 4xx (e.g., 422 for malformed request): log the response body, treat as final failure, zero-vector fallback.
- HTTP 5xx: retry with backoff.
- Server completely down: precompute aborts with a clear error message pointing to the BGE-M3 endpoint URL. User should verify the server is running.

## Testing

- `tests/test_honeypot.py` and `tests/test_scoring.py` do not touch embeddings, so they should still pass.
- Manual smoke test:
  1. Verify BGE-M3 endpoint is reachable: `curl -X POST http://192.168.31.246:5000/embed -H "Content-Type: application/json" -d '{"inputs":["hello world"]}'` returns `[[0.1, ...]]` (a list of one 1024-dim vector).
  2. Run `python precompute.py --candidates ./sample/candidates.jsonl --jd ./sample/job_description.txt --out ./artifacts/` and confirm it completes without Gemini API key.
  3. Run `python rank.py --candidates ./sample/candidates.jsonl --out ./submission.csv` and confirm the CSV is produced with 100 rows.
  4. Run `pytest tests/ -v` and confirm existing tests pass.
- New unit test (optional but recommended): `tests/test_retrieval.py` that mocks `requests.post` and asserts `compute_bge_m3_embeddings` returns a `(N, 1024)` array.

## Out of Scope

- Migrating the re-ranker from DeepSeek to anything else.
- Refactoring the embedding logic into an abstract `EmbeddingClient` interface.
- Changing the FAISS index type (still `IndexFlatIP`).
- Changes to scoring, reasoning, or honeypot logic.

## Success Criteria

- `precompute.py` runs end-to-end without a `GOOGLE_API_KEY` env var.
- `embeddings.npy` has shape `(N, 1024)` for N candidates.
- `query_embedding.npy` has shape `(1024,)`.
- `submission.csv` has 100 rows with the expected schema.
- No `import google.genai` statements remain in the source tree.
