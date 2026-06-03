import pickle
from pathlib import Path
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    return [token for token in "".join(ch if ch.isalnum() else " " for ch in text.lower()).split() if token]


def build_bm25_index(texts: List[str]) -> BM25Okapi:
    tokenized = [_tokenize(text) for text in texts]
    return BM25Okapi(tokenized)


def save_bm25_index(index: BM25Okapi, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(index, handle)


def build_faiss_index(
    embeddings: np.ndarray,
    nlist: int,
    nprobe: int,
    seed: int = 42,
):
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("faiss-cpu is required for FAISS indexing") from exc

    dim = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    faiss.seed_random(seed)

    sample_size = min(len(embeddings), max(10000, nlist * 5))
    if sample_size < len(embeddings):
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(len(embeddings), size=sample_size, replace=False)
        train_vectors = embeddings[sample_idx]
    else:
        train_vectors = embeddings

    index.train(train_vectors.astype(np.float32))
    index.add(embeddings.astype(np.float32))
    index.nprobe = nprobe
    return index


def save_faiss_index(index: "faiss.Index", path: Path) -> None:
    import faiss

    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
