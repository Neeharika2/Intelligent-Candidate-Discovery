"""Precompute pipeline for offline artifacts."""

from .loader import load_candidates
from .feature_engineer import build_feature_frame
from .honeypot_detector import detect_honeypots
from .embedder import GeminiEmbedder, embed_candidate_profiles, embed_candidate_skills
from .jd_embedder import embed_jd
from .indexer import build_faiss_index, build_bm25_index
from .embedding_cache import save_embeddings, load_embeddings

__all__ = [
    "load_candidates",
    "build_feature_frame",
    "detect_honeypots",
    "GeminiEmbedder",
    "embed_candidate_profiles",
    "embed_candidate_skills",
    "embed_jd",
    "build_faiss_index",
    "build_bm25_index",
    "save_embeddings",
    "load_embeddings",
]
