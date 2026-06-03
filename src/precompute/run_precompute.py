import json
import time
from pathlib import Path
from typing import Optional

from src.config import (
    EMBED_BATCH_SIZE,
    EMBED_MODEL,
    FAISS_NLIST,
    FAISS_NPROBE,
    GEMINI_API_KEY,
    JD_EMBEDDINGS_PATH,
    PRECOMPUTED_DIR,
    SAMPLE_JSONL,
    SCHEMA_PATH,
    CANDIDATE_EMBEDDINGS_PATH,
    SKILL_EMBEDDINGS_PATH,
    FEATURE_PARQUET_PATH,
    HONEYPOT_FLAGS_PATH,
    CANDIDATE_IDS_PATH,
    FAISS_INDEX_PATH,
    BM25_INDEX_PATH,
)
from src.precompute.embedding_cache import save_embeddings
from src.precompute.embedder import GeminiEmbedder, embed_candidate_profiles, embed_candidate_skills
from src.precompute.feature_engineer import build_feature_frame
from src.precompute.honeypot_detector import detect_honeypots
from src.precompute.indexer import build_bm25_index, build_faiss_index, save_bm25_index, save_faiss_index
from src.precompute.jd_embedder import embed_jd
from src.precompute.loader import load_candidates
from src.utils.logging import StageLogger


def run_precompute(
    candidates_path: Optional[Path] = None,
    schema_path: Optional[Path] = None,
    jd_text_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    max_records: Optional[int] = None,
) -> None:
    output_dir = output_dir or PRECOMPUTED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_path = output_dir / FEATURE_PARQUET_PATH.name
    candidate_embed_path = output_dir / CANDIDATE_EMBEDDINGS_PATH.name
    skill_embed_path = output_dir / SKILL_EMBEDDINGS_PATH.name
    honeypot_path = output_dir / HONEYPOT_FLAGS_PATH.name
    candidate_ids_path = output_dir / CANDIDATE_IDS_PATH.name
    faiss_path = output_dir / FAISS_INDEX_PATH.name
    bm25_path = output_dir / BM25_INDEX_PATH.name
    jd_embed_path = output_dir / JD_EMBEDDINGS_PATH.name

    logger = StageLogger(output_dir / "precompute_log.jsonl")

    candidates_path = candidates_path or SAMPLE_JSONL
    schema_path = schema_path or SCHEMA_PATH

    start = time.time()
    candidates, invalid = load_candidates(candidates_path, schema_path, max_records=max_records)
    logger.log_stage(
        "load_candidates",
        time.time() - start,
        candidates_count=len(candidates),
        metadata={"invalid_records": len(invalid)},
    )

    start = time.time()
    feature_frame = build_feature_frame(candidates)
    feature_frame.to_parquet(feature_path, index=False)
    logger.log_stage(
        "feature_engineer",
        time.time() - start,
        candidates_count=len(feature_frame),
    )

    start = time.time()
    honeypot_flags = detect_honeypots(candidates)
    with open(honeypot_path, "w", encoding="utf-8") as handle:
        json.dump(honeypot_flags, handle, indent=2)
    logger.log_stage(
        "honeypot_detection",
        time.time() - start,
        candidates_count=len(honeypot_flags),
    )

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set; embeddings require network access")

    embedder = GeminiEmbedder(
        api_key=GEMINI_API_KEY,
        model_name=EMBED_MODEL,
        batch_size=EMBED_BATCH_SIZE,
    )

    start = time.time()
    profile_embeddings = embed_candidate_profiles(embedder, candidates)
    save_embeddings(
        candidate_embed_path,
        profile_embeddings,
        model_name=EMBED_MODEL,
        embedding_dim=profile_embeddings.shape[1],
    )
    logger.log_stage(
        "profile_embeddings",
        time.time() - start,
        candidates_count=len(profile_embeddings),
    )

    start = time.time()
    skill_embeddings = embed_candidate_skills(embedder, candidates)
    save_embeddings(
        skill_embed_path,
        skill_embeddings,
        model_name=EMBED_MODEL,
        embedding_dim=skill_embeddings.shape[1],
    )
    logger.log_stage(
        "skill_embeddings",
        time.time() - start,
        candidates_count=len(skill_embeddings),
    )

    if jd_text_path:
        with open(jd_text_path, "r", encoding="utf-8") as handle:
            jd_text = handle.read().strip()
        if jd_text:
            start = time.time()
            embed_jd(embedder, jd_text, jd_embed_path)
            logger.log_stage("jd_embedding", time.time() - start)

    start = time.time()
    faiss_index = build_faiss_index(profile_embeddings, FAISS_NLIST, FAISS_NPROBE)
    save_faiss_index(faiss_index, faiss_path)
    logger.log_stage("faiss_index", time.time() - start)

    start = time.time()
    bm25_index = build_bm25_index(feature_frame["profile_text"].tolist())
    save_bm25_index(bm25_index, bm25_path)
    logger.log_stage("bm25_index", time.time() - start)

    candidate_ids = feature_frame["candidate_id"].tolist()
    with open(candidate_ids_path, "w", encoding="utf-8") as handle:
        json.dump(candidate_ids, handle, indent=2)

    logger.log_stage("precompute_complete", 0.0, candidates_count=len(candidate_ids))
