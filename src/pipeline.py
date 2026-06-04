import json
import pickle
import os
import numpy as np
import pandas as pd

from src.loader import load_candidates, build_retrieval_corpus, get_candidate_id
from src.honeypot import detect_honeypot
from src.scoring import combine_scores, tier_normalize
from src.reasoning import generate_reasoning
from src.retrieval import hybrid_retrieve, tokenize
from src.reranking import rerank_candidates, combine_final_score
from src.config import RERANK_TOP_N, BGE_M3_API_URL, BGE_M3_REQUEST_TIMEOUT


def run_precompute(
    candidates_path: str,
    jd_path: str,
    output_dir: str,
    deepseek_api_key: str = None,
):
    os.makedirs(output_dir, exist_ok=True)

    ds_key = deepseek_api_key

    candidates = load_candidates(candidates_path)
    jd_text = open(jd_path, "r", encoding="utf-8").read()

    corpus = build_retrieval_corpus(candidates)

    ideal_profile_path = os.path.join(output_dir, "ideal_profile.txt")
    if os.path.exists(ideal_profile_path):
        print("Loading ideal profile from disk...")
        try:
            with open(ideal_profile_path, "r", encoding="utf-8") as f:
                ideal_profile = f.read()
        except UnicodeDecodeError:
            with open(ideal_profile_path, "r", encoding="cp1252") as f:
                ideal_profile = f.read()
    else:
        from src.query import extract_ideal_profile
        ideal_profile = extract_ideal_profile(jd_text, api_key=ds_key)
        with open(ideal_profile_path, "w", encoding="utf-8") as f:
            f.write(ideal_profile)

    from src.retrieval import build_bm25_index_save, compute_bge_m3_embeddings, build_faiss_index_save
    bm25_path = os.path.join(output_dir, "bm25_index.pkl")
    if os.path.exists(bm25_path):
        print("Loading BM25 index from disk...")
        with open(bm25_path, "rb") as f:
            _, bm25 = pickle.load(f)
    else:
        bm25 = build_bm25_index_save(corpus, bm25_path)

    embedding_checkpoint = os.path.join(output_dir, "embedding_checkpoint.npz")
    embeddings = compute_bge_m3_embeddings(
        corpus, checkpoint_path=embedding_checkpoint
    )
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)

    import requests
    query_resp = requests.post(
        BGE_M3_API_URL,
        json={"inputs": [ideal_profile]},
        timeout=BGE_M3_REQUEST_TIMEOUT,
    )
    query_resp.raise_for_status()
    query_embedding = np.array(query_resp.json()[0], dtype=np.float32)
    np.save(os.path.join(output_dir, "query_embedding.npy"), query_embedding)

    faiss_index = build_faiss_index_save(embeddings, os.path.join(output_dir, "faiss.index"))

    honeypots = {}
    for c in candidates:
        flagged, reason = detect_honeypot(c)
        if flagged:
            honeypots[c["candidate_id"]] = reason
    with open(os.path.join(output_dir, "honeypots.json"), "w") as f:
        json.dump(honeypots, f)

    feature_data = []
    for i, c in enumerate(candidates):
        feature_score, evidence = combine_scores(c)
        feature_data.append({
            "candidate_id": c["candidate_id"],
            "feature_score": feature_score,
            "evidence": json.dumps(evidence),
        })
    pd.DataFrame(feature_data).to_parquet(os.path.join(output_dir, "features.parquet"))

    candidate_titles = [c.get("profile", {}).get("current_title", "") for c in candidates]

    top_n_indices = _get_top_n_by_feature(candidates, feature_data, RERANK_TOP_N)
    top_n_candidates = [candidates[i] for i in top_n_indices]

    rerank_scores, rerank_reasoning = rerank_candidates(
        top_n_candidates,
        {c["candidate_id"]: feature_data[i]["feature_score"] for i, c in enumerate(candidates)},
        ideal_profile,
        api_key=ds_key,
        checkpoint_path=os.path.join(output_dir, "rerank_progress.json"),
    )
    with open(os.path.join(output_dir, "rerank_scores.json"), "w") as f:
        json.dump(rerank_scores, f)
    with open(os.path.join(output_dir, "rerank_reasoning.json"), "w") as f:
        json.dump(rerank_reasoning, f)

    with open(os.path.join(output_dir, "candidate_titles.json"), "w") as f:
        json.dump(candidate_titles, f)

    print(f"Pre-computation complete. Artifacts saved to {output_dir}")
    print(f"Total candidates: {len(candidates)}")
    print(f"Honeypots detected: {len(honeypots)}")
    print(f"Re-ranked top {len(top_n_candidates)} candidates")


def _get_top_n_by_feature(candidates, feature_data, n):
    scored = [(i, feature_data[i]["feature_score"]) for i in range(len(candidates))]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored[:n]]


def run_ranking(
    artifacts_dir: str,
    candidates_path: str,
    output_csv: str,
):
    candidates = load_candidates(candidates_path)

    with open(os.path.join(artifacts_dir, "ideal_profile.txt"), "r") as f:
        ideal_profile = f.read()

    with open(os.path.join(artifacts_dir, "bm25_index.pkl"), "rb") as f:
        tokenized_corpus, bm25 = pickle.load(f)

    embeddings = np.load(os.path.join(artifacts_dir, "embeddings.npy"))
    faiss_index_path = os.path.join(artifacts_dir, "faiss.index")
    import faiss
    faiss_index = faiss.read_index(faiss_index_path)

    query_embedding = np.load(os.path.join(artifacts_dir, "query_embedding.npy"))

    with open(os.path.join(artifacts_dir, "honeypots.json"), "r") as f:
        honeypots = json.load(f)

    features_df = pd.read_parquet(os.path.join(artifacts_dir, "features.parquet"))

    with open(os.path.join(artifacts_dir, "rerank_scores.json"), "r") as f:
        rerank_scores = json.load(f)
    with open(os.path.join(artifacts_dir, "rerank_reasoning.json"), "r") as f:
        rerank_reasoning = json.load(f)
    with open(os.path.join(artifacts_dir, "candidate_titles.json"), "r") as f:
        candidate_titles = json.load(f)

    query_tokens = tokenize(ideal_profile)
    retrieved_indices = hybrid_retrieve(
        bm25, faiss_index, query_tokens, query_embedding,
        candidate_titles
    )

    all_scores = {}
    all_evidence = {}
    for _, row in features_df.iterrows():
        cid = row["candidate_id"]
        all_scores[cid] = row["feature_score"]
        all_evidence[cid] = json.loads(row["evidence"]) if isinstance(row["evidence"], str) else row["evidence"]

    scored_candidates = []
    for idx in retrieved_indices:
        c = candidates[idx]
        cid = c["candidate_id"]
        if cid in honeypots:
            continue

        feature_score = all_scores.get(cid, 0.0)
        evidence = all_evidence.get(cid, {})
        final_score = combine_final_score(cid, feature_score, rerank_scores)
        llm_reason = rerank_reasoning.get(cid, None)
        reasoning = generate_reasoning(c, evidence, llm_reason)

        scored_candidates.append({
            "candidate_id": cid,
            "score": final_score,
            "reasoning": reasoning,
        })

    scored_candidates.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    all_blended_scores = [sc["score"] for sc in scored_candidates]
    for i, sc in enumerate(scored_candidates):
        raw = sc["score"]
        sc["score"] = tier_normalize(raw, all_blended_scores if len(all_blended_scores) > 0 else None)

    scored_candidates = scored_candidates[:100]

    rows = []
    for rank, sc in enumerate(scored_candidates, 1):
        rows.append({
            "candidate_id": sc["candidate_id"],
            "rank": rank,
            "score": round(sc["score"], 4),
            "reasoning": sc["reasoning"],
        })

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Submission written to {output_csv}")
    print(f"Top candidate: {rows[0]['candidate_id']} (score={rows[0]['score']})")
    print(f"Bottom candidate: {rows[-1]['candidate_id']} (score={rows[-1]['score']})")
    return rows