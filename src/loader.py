import json
import gzip


def load_candidates(path: str) -> list[dict]:
    if path.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    elif path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return [data]
    else:
        raise ValueError(f"Unsupported file format: {path}")


def build_retrieval_corpus(candidates: list[dict]) -> list[str]:
    corpus = []
    for c in candidates:
        parts = []
        profile = c.get("profile", {})
        if profile.get("headline"):
            parts.append(profile["headline"])
        if profile.get("summary"):
            parts.append(profile["summary"])
        for h in c.get("career_history", []):
            if h.get("description"):
                parts.append(h["description"])
        corpus.append(" ".join(parts))
    return corpus


def get_candidate_id(idx: int, candidates: list[dict]) -> str:
    return candidates[idx].get("candidate_id", f"UNK_{idx}")


def get_candidate_text_for_reranking(c: dict) -> str:
    parts = []
    profile = c.get("profile", {})
    parts.append(f"Title: {profile.get('current_title', 'N/A')} at {profile.get('current_company', 'N/A')} ({profile.get('current_industry', 'N/A')}, {profile.get('current_company_size', 'N/A')})")
    parts.append(f"Experience: {profile.get('years_of_experience', 0):.1f}yr")
    parts.append(f"Location: {profile.get('location', 'N/A')}, {profile.get('country', 'N/A')}")
    for h in c.get("career_history", []):
        parts.append(f"- {h.get('title', 'N/A')} at {h.get('company', 'N/A')} ({h.get('duration_months', 0)}mo): {h.get('description', '')}")
    skill_names = [s["name"] for s in c.get("skills", [])]
    if skill_names:
        parts.append(f"Skills: {', '.join(skill_names)}")
    signals = c.get("redrob_signals", {})
    parts.append(f"Response rate: {signals.get('recruiter_response_rate', 'N/A')}")
    parts.append(f"Last active: {signals.get('last_active_date', 'N/A')}")
    parts.append(f"Notice period: {signals.get('notice_period_days', 'N/A')}d")
    parts.append(f"GitHub score: {signals.get('github_activity_score', 'N/A')}")
    parts.append(f"Open to work: {signals.get('open_to_work_flag', 'N/A')}")
    return "\n".join(parts)