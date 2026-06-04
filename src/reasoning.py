from src.config import WEIGHTS


def generate_reasoning(candidate: dict, feature_evidence: dict, llm_reasoning: str = None) -> str:
    if llm_reasoning:
        if len(llm_reasoning) > 200:
            return llm_reasoning[:197] + "..."
        return llm_reasoning

    parts = []
    profile = candidate.get("profile", {})

    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "")
    industry = profile.get("current_industry", "")
    yoe = profile.get("years_of_experience", 0)

    parts.append(f"{title} at {company} ({industry}), {yoe:.1f}yr")

    sorted_features = sorted(
        [(k, v) for k, v in feature_evidence.items() if isinstance(v, tuple) and len(v) >= 3 and v[0] > 0],
        key=lambda x: x[1][0],
        reverse=True,
    )[:3]

    for name, (score, max_score, evidence) in sorted_features:
        if evidence:
            parts.append(evidence)

    signals = candidate.get("redrob_signals", {})
    country = profile.get("country", "")
    location = profile.get("location", "")
    notice = signals.get("notice_period_days", "N/A")
    resp_rate = signals.get("recruiter_response_rate", 0)

    if country == "India":
        parts.append(f"based in {location}")
    else:
        parts.append(f"based in {location}, {country}")

    if notice and notice <= 60:
        parts.append(f"{notice}d notice")
    if resp_rate > 0.5:
        parts.append(f"response rate {resp_rate:.0%}")

    return "; ".join(parts) + "."