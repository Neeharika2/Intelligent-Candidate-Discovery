import json
import os
import time
import re

from openai import OpenAI

from src.config import RERANK_MODEL, RERANK_TOP_N, RERANK_BATCH_SAVE_EVERY
from src.query import build_rerank_prompt


def _get_client(api_key: str = None) -> OpenAI:
    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    return OpenAI(api_key=key, base_url="https://api.deepseek.com/v1")


def rerank_candidates(
    candidates: list[dict],
    feature_scores: dict[str, float],
    ideal_profile: str,
    api_key: str = None,
    checkpoint_path: str = None,
) -> tuple[dict[str, float], dict[str, str]]:
    client = _get_client(api_key)
    rerank_scores = {}
    rerank_reasoning = {}

    start_idx = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            progress = json.load(f)
        rerank_scores = progress.get("scores", {})
        rerank_reasoning = progress.get("reasoning", {})
        start_idx = progress.get("last_index", 0)

    from src.loader import get_candidate_text_for_reranking

    for i in range(start_idx, len(candidates)):
        c = candidates[i]
        cid = c.get("candidate_id", f"idx_{i}")
        cand_text = get_candidate_text_for_reranking(c)
        prompt = build_rerank_prompt(ideal_profile, cand_text)

        retries = 0
        success = False
        while retries < 5 and not success:
            try:
                response = client.chat.completions.create(
                    model=RERANK_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1000,
                    timeout=60.0,
                )
                text = response.choices[0].message.content.strip()
                match = re.search(r'\{[^}]+\}', text, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    score = int(data.get("score", 3))
                    reasoning = data.get("reasoning", "")
                else:
                    score = 3
                    reasoning = "Could not parse LLM response"

                rerank_scores[cid] = score
                rerank_reasoning[cid] = reasoning
                success = True
            except Exception as e:
                retries += 1
                wait_time = min(2 ** retries, 60)
                print(f"Re-rank failed for {cid} ({e}), retrying in {wait_time}s...")
                time.sleep(wait_time)

        if not success:
            rerank_scores[cid] = 3
            rerank_reasoning[cid] = "LLM call failed, using neutral score"

        if (i + 1) % RERANK_BATCH_SAVE_EVERY == 0 and checkpoint_path:
            with open(checkpoint_path, "w") as f:
                json.dump({
                    "scores": rerank_scores,
                    "reasoning": rerank_reasoning,
                    "last_index": i + 1,
                }, f)

    if checkpoint_path:
        with open(checkpoint_path, "w") as f:
            json.dump({
                "scores": rerank_scores,
                "reasoning": rerank_reasoning,
                "last_index": len(candidates),
            }, f)

    return rerank_scores, rerank_reasoning


def combine_final_score(
    candidate_id: str,
    feature_score: float,
    rerank_scores: dict[str, float],
) -> float:
    if candidate_id in rerank_scores:
        llm_normalized = rerank_scores[candidate_id] / 5.0
        return 0.4 * llm_normalized + 0.6 * feature_score
    return feature_score