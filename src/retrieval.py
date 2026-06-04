import pickle
import re
import numpy as np
import json
import os
import time
from tqdm import tqdm

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

import requests

from src.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_CHECKPOINT_EVERY,
    BGE_M3_API_URL,
    BGE_M3_EMBEDDING_DIM,
    BGE_M3_REQUEST_TIMEOUT,
    BGE_M3_MAX_RETRIES,
    RERANK_MODEL,
    RERANK_TOP_N,
    RERANK_BATCH_SAVE_EVERY,
)


def tokenize(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())


def build_bm25_index(corpus: list[str]) -> BM25Okapi:
    tokenized = [tokenize(doc) for doc in corpus]
    return BM25Okapi(tokenized)


def build_bm25_index_save(corpus: list[str], path: str) -> BM25Okapi:
    bm25 = build_bm25_index(corpus)
    tokenized = [tokenize(doc) for doc in corpus]
    with open(path, "wb") as f:
        pickle.dump((tokenized, bm25), f)
    return bm25


def compute_bge_m3_embeddings(
    texts: list[str],
    checkpoint_path: str = None,
) -> np.ndarray:
    all_embeddings = [None] * len(texts)

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with np.load(checkpoint_path, allow_pickle=True) as data:
                if "embeddings" in data:
                    saved_embeds = data["embeddings"]
                    if saved_embeds.ndim == 2 and saved_embeds.shape[1] != BGE_M3_EMBEDDING_DIM:
                        raise ValueError(
                            f"Checkpoint dim mismatch: saved {saved_embeds.shape[1]}, "
                            f"expected {BGE_M3_EMBEDDING_DIM}. "
                            f"Delete {checkpoint_path} and re-run."
                        )
                    loaded_count = 0
                    for i in range(min(len(saved_embeds), len(texts))):
                        emb = saved_embeds[i]
                        if emb is not None and not np.all(emb == 0.0):
                            all_embeddings[i] = list(emb)
                            loaded_count += 1
                    print(f"Loaded {loaded_count} existing embeddings from checkpoint.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")

    needed_indices = [i for i, emb in enumerate(all_embeddings) if emb is None]

    batches = []
    for i in range(0, len(needed_indices), EMBEDDING_BATCH_SIZE):
        batch_idxs = needed_indices[i:i + EMBEDDING_BATCH_SIZE]
        batch_texts = [texts[idx] for idx in batch_idxs]
        batches.append((batch_idxs, batch_texts))

    def embed_batch(item):
        batch_idxs, batch_texts = item
        retries = 0
        while retries < BGE_M3_MAX_RETRIES:
            try:
                resp = requests.post(
                    BGE_M3_API_URL,
                    json={"inputs": batch_texts},
                    timeout=BGE_M3_REQUEST_TIMEOUT,
                )
                if 400 <= resp.status_code < 500:
                    print(
                        f"BGE-M3 returned {resp.status_code} (client error, not retrying): "
                        f"{resp.text[:500]}"
                    )
                    return batch_idxs, [[0.0] * BGE_M3_EMBEDDING_DIM for _ in batch_texts]
                resp.raise_for_status()
                vectors = resp.json()
                if not isinstance(vectors, list) or len(vectors) != len(batch_texts):
                    raise ValueError(
                        f"Unexpected response shape: got {type(vectors).__name__} "
                        f"with {len(vectors) if isinstance(vectors, list) else '?'} items, "
                        f"expected {len(batch_texts)}"
                    )
                return batch_idxs, vectors
            except (requests.exceptions.RequestException, OSError, ValueError, json.JSONDecodeError) as e:
                retries += 1
                wait_time = min(2 ** retries, 60)
                print(f"Embedding batch failed ({e}), retrying in {wait_time}s...")
                time.sleep(wait_time)
        return batch_idxs, [[0.0] * BGE_M3_EMBEDDING_DIM for _ in batch_texts]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    max_workers = 5
    print(f"Starting concurrent embedding with {max_workers} threads. Needed batches: {len(batches)}...")

    original_total_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
    completed_batches = original_total_batches - len(batches)

    if len(batches) > 0:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(embed_batch, item): item for item in batches}
            pbar = tqdm(total=original_total_batches, initial=completed_batches, desc="Embedding (BGE-M3)")

            batch_counter = 0
            for future in as_completed(futures):
                batch_idxs, batch_embeds = future.result()
                for idx, embed in zip(batch_idxs, batch_embeds):
                    all_embeddings[idx] = embed

                pbar.update(1)
                batch_counter += 1

                if checkpoint_path and batch_counter % EMBEDDING_CHECKPOINT_EVERY == 0:
                    first_none = 0
                    while first_none < len(texts) and all_embeddings[first_none] is not None:
                        first_none += 1

                    save_arr = np.zeros((len(texts), BGE_M3_EMBEDDING_DIM), dtype=np.float32)
                    for idx, emb in enumerate(all_embeddings):
                        if emb is not None:
                            save_arr[idx] = emb
                    np.savez(checkpoint_path, embeddings=save_arr, done_count=first_none)

            pbar.close()

    result = np.zeros((len(texts), BGE_M3_EMBEDDING_DIM), dtype=np.float32)
    for idx, emb in enumerate(all_embeddings):
        if emb is not None:
            result[idx] = emb

    if checkpoint_path:
        first_none = 0
        while first_none < len(texts) and all_embeddings[first_none] is not None:
            first_none += 1
        np.savez(checkpoint_path, embeddings=result, done_count=first_none)
    return result


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def build_faiss_index_save(embeddings: np.ndarray, path: str) -> faiss.IndexFlatIP:
    index = build_faiss_index(embeddings)
    faiss.write_index(index, path)
    return index


def hybrid_retrieve(
    bm25: BM25Okapi,
    faiss_index: faiss.IndexFlatIP,
    query_bm25_tokens: list[str],
    query_embedding: np.ndarray,
    candidate_titles: list[str],
    top_k_bm25: int = 2000,
    top_k_faiss: int = 2000,
    k: int = 60,
) -> list[int]:
    from src.config import ALL_RELEVANT_SKILLS

    bm25_scores = bm25.get_scores(query_bm25_tokens)
    bm25_top_indices = np.argsort(bm25_scores)[-top_k_bm25:][::-1]

    query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(query_vec)
    faiss_scores, faiss_top_indices = faiss_index.search(query_vec, top_k_faiss)
    faiss_top = faiss_top_indices[0]

    rrf_scores = {}
    for rank, idx in enumerate(bm25_top_indices):
        idx = int(idx)
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)

    for rank, idx in enumerate(faiss_top):
        idx = int(idx)
        if idx >= 0:
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)

    ml_keywords = {"ml", "ai", "machine learning", "data scientist", "data engineer",
                   "nlp", "deep learning", "retrieval", "search", "ranking"}
    for i, title in enumerate(candidate_titles):
        t = (title or "").lower()
        if any(kw in t for kw in ml_keywords):
            if i not in rrf_scores:
                rrf_scores[i] = 1.0 / (k + 3000)

    sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return sorted_indices


def compute_bm25_scores(bm25: BM25Okapi, query_tokens: list[str]) -> np.ndarray:
    return bm25.get_scores(query_tokens)