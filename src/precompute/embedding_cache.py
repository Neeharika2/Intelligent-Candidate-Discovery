import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.utils.logging import calculate_checksum


def _metadata_path(path: Path) -> Path:
    return Path(str(path) + ".meta.json")


def save_embeddings(
    path: Path,
    embeddings: np.ndarray,
    model_name: str,
    embedding_dim: int,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings.astype(np.float32))

    metadata = {
        "model_name": model_name,
        "embedding_dim": embedding_dim,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checksum": calculate_checksum(path),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    meta_path = _metadata_path(path)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return metadata


def load_embeddings(
    path: Path,
    expected_model_name: Optional[str] = None,
    expected_dim: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    meta_path = _metadata_path(path)
    with open(meta_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if expected_model_name and metadata.get("model_name") != expected_model_name:
        raise ValueError("Embedding model mismatch")
    if expected_dim and metadata.get("embedding_dim") != expected_dim:
        raise ValueError("Embedding dimension mismatch")

    embeddings = np.load(path)
    checksum = calculate_checksum(path)
    if metadata.get("checksum") and metadata.get("checksum") != checksum:
        raise ValueError("Embedding checksum mismatch")

    return embeddings, metadata
