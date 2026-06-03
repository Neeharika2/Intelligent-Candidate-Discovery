import time
from typing import Any, Dict, List, Tuple

import numpy as np

from src.precompute.feature_engineer import build_profile_chunks, build_skills_text


class GeminiEmbedder:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        batch_size: int = 64,
        max_retries: int = 4,
    ) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError("google-genai is required for embedding") from exc

        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._batch_size = batch_size
        self._max_retries = max_retries

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vectors: List[List[float]] = []
        for batch_start in range(0, len(texts), self._batch_size):
            batch = texts[batch_start : batch_start + self._batch_size]
            response = self._retry_embed(batch)
            vectors.extend(response)

        arr = np.asarray(vectors, dtype=np.float32)
        return _normalize_vectors(arr)

    def _retry_embed(self, texts: List[str]) -> List[List[float]]:
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.models.embed_content(
                    model=self._model_name,
                    contents=texts,
                )
                return [item.values for item in response.embeddings]
            except Exception:
                if attempt >= self._max_retries:
                    raise
                time.sleep(2 ** attempt)
        return []

    @property
    def batch_size(self) -> int:
        return self._batch_size


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _weighted_average(vectors: np.ndarray, weights: List[float]) -> np.ndarray:
    weight_arr = np.asarray(weights, dtype=np.float32)
    if weight_arr.sum() == 0:
        weight_arr = np.ones_like(weight_arr)
    weighted = (vectors.T * weight_arr).T
    return weighted.sum(axis=0) / weight_arr.sum()


def embed_candidate_profiles(
    embedder: GeminiEmbedder,
    candidates: List[Dict[str, Any]],
) -> np.ndarray:
    candidate_sums: List[np.ndarray] = [None] * len(candidates)
    candidate_weights = [0.0] * len(candidates)

    batch_texts: List[str] = []
    batch_weights: List[float] = []
    batch_ids: List[int] = []

    def flush_batch() -> None:
        if not batch_texts:
            return
        vectors = embedder.embed_texts(batch_texts)
        for idx, vector in enumerate(vectors):
            candidate_idx = batch_ids[idx]
            weight = batch_weights[idx]
            if candidate_sums[candidate_idx] is None:
                candidate_sums[candidate_idx] = vector * weight
            else:
                candidate_sums[candidate_idx] += vector * weight
            candidate_weights[candidate_idx] += weight

        batch_texts.clear()
        batch_weights.clear()
        batch_ids.clear()

    for candidate_idx, candidate in enumerate(candidates):
        chunks = build_profile_chunks(candidate)
        if not chunks:
            chunks = [(" ", 1.0)]
        for text, weight in chunks:
            batch_texts.append(text)
            batch_weights.append(weight)
            batch_ids.append(candidate_idx)
            if len(batch_texts) >= embedder.batch_size:
                flush_batch()

    flush_batch()

    dim = None
    for vector in candidate_sums:
        if vector is not None:
            dim = vector.shape[0]
            break

    if dim is None:
        return np.zeros((len(candidates), 1), dtype=np.float32)

    embeddings: List[np.ndarray] = []
    for idx in range(len(candidates)):
        if candidate_sums[idx] is None:
            embeddings.append(np.zeros((dim,), dtype=np.float32))
            continue
        if candidate_weights[idx] == 0:
            embeddings.append(candidate_sums[idx])
        else:
            embeddings.append(candidate_sums[idx] / candidate_weights[idx])

    return _normalize_vectors(np.vstack(embeddings))


def embed_candidate_skills(
    embedder: GeminiEmbedder,
    candidates: List[Dict[str, Any]],
) -> np.ndarray:
    texts = [build_skills_text(candidate.get("skills", [])) for candidate in candidates]
    return embedder.embed_texts(texts)
