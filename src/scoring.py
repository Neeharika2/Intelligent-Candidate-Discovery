import math
from src.config import (
    ALL_RELEVANT_SKILLS, CORE_ML_SKILLS, RETRIEVAL_SKILLS, EMBEDDING_SKILLS,
    VECTOR_DB_SKILLS, LLM_SKILLS, NLP_SKILLS, PYTHON_SKILLS, WRONG_DOMAIN_SKILLS,
    WEIGHTS, CONSULTING_INDUSTRIES, PREFERRED_CITIES, TIER1_INDIAN_CITIES,
    POSITIVE_CAREER_PHRASES, NEGATIVE_CAREER_PHRASES, SCORE_TIERS,
    normalize_skill_name, normalize_city, skill_matches_taxonomy,
    count_taxonomy_skills, gaussian_score,
)


def score_technical_fit(candidate: dict) -> tuple[float, dict]:
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})
    evidence = {}

    core_ml = count_taxonomy_skills(skills, CORE_ML_SKILLS)
    retrieval = count_taxonomy_skills(skills, RETRIEVAL_SKILLS)
    embedding = count_taxonomy_skills(skills, EMBEDDING_SKILLS)
    vector_db = count_taxonomy_skills(skills, VECTOR_DB_SKILLS)
    llm = count_taxonomy_skills(skills, LLM_SKILLS)
    nlp = count_taxonomy_skills(skills, NLP_SKILLS)

    skill_core = min(core_ml + retrieval + embedding, 8.0)
    skill_vecdb = min(vector_db, 5.0)
    skill_llm = min(llm, 4.0)
    skill_nlp = min(nlp, 4.0)

    assessments = signals.get("skill_assessment_scores", {})
    for s in skills:
        if s.get("proficiency") == "expert" and s.get("name") in assessments:
            if assessments[s["name"]] < 40:
                category = None
                for tax in [CORE_ML_SKILLS, RETRIEVAL_SKILLS, EMBEDDING_SKILLS]:
                    if skill_matches_taxonomy(s["name"], tax):
                        category = "core"
                        break
                if not category:
                    for tax in [VECTOR_DB_SKILLS]:
                        if skill_matches_taxonomy(s["name"], tax):
                            category = "vecdb"
                            break
                if category == "core":
                    skill_core *= 0.5
                elif category == "vecdb":
                    skill_vecdb *= 0.5

    skill_python = 0.0
    career_text = " ".join(h.get("description", "") for h in career).lower()
    summary_text = (profile.get("summary", "") or "").lower()
    all_text = career_text + " " + summary_text

    has_python_skill = any(
        skill_matches_taxonomy(s["name"], PYTHON_SKILLS) for s in skills
    )
    if has_python_skill:
        skill_python = 3.0
    elif "python" in all_text:
        skill_python = 2.0

    positive_count = sum(1 for p in POSITIVE_CAREER_PHRASES if p in all_text)
    career_positive = min(positive_count * 1.6, 8.0)

    negative_count = sum(1 for p in NEGATIVE_CAREER_PHRASES if p in all_text)
    has_positive = positive_count > 0
    career_negative = min(negative_count * 2.0, 6.0)
    if has_positive:
        career_negative *= 0.3  # reduce but don't eliminate when positives exist

    github = signals.get("github_activity_score", -1)
    if github == -1:
        github_score = 0.0
    elif github < 30:
        github_score = 1.0
    elif github < 60:
        github_score = 2.0
    else:
        github_score = 3.0

    raw = skill_core + skill_vecdb + skill_llm + skill_nlp + skill_python + career_positive - career_negative + github_score
    raw = max(raw, 0.0)
    max_raw = 35.0

    evidence["skill_core_ml"] = (skill_core, 8.0, f"{core_ml:.1f} ML/retrieval/embedding skill points")
    evidence["skill_vecdb"] = (skill_vecdb, 5.0, f"{vector_db:.1f} vector DB skill points")
    evidence["career_positive"] = (career_positive, 8.0, f"{positive_count} positive career phrases")
    evidence["career_negative"] = (-career_negative, 0.0, f"{negative_count} negative career phrases (deducted)")
    evidence["github"] = (github_score, 3.0, f"github score {github}")

    return raw / max_raw, evidence


def score_experience_fit(candidate: dict) -> tuple[float, dict]:
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])
    evidence = {}

    yoe = profile.get("years_of_experience", 0)
    yoe_score = gaussian_score(yoe, center=7, sigma=2.5, max_score=8.0)

    total_months = sum(h.get("duration_months", 0) for h in career)
    non_consulting_months = sum(
        h.get("duration_months", 0) for h in career
        if h.get("industry", "") not in CONSULTING_INDUSTRIES
    )
    product_company_pct = non_consulting_months / total_months if total_months > 0 else 1.0
    product_score = product_company_pct * 7.0

    ml_titles = {"ml engineer", "machine learning engineer", "ai engineer",
                 "data scientist", "data engineer", "ml researcher",
                 "senior ml engineer", "senior ai engineer", "staff ml engineer"}
    title_score = 0.0
    for idx, h in enumerate(career):
        t = (h.get("title", "") or "").lower()
        if any(kw in t for kw in ml_titles):
            recency_weight = 1.0 - (idx * 0.15)  # first entry = most recent
            recency_weight = max(recency_weight, 0.3)
            title_score = max(title_score, 5.0 * recency_weight)

    startup_score = 0.0
    for h in career:
        sz = h.get("company_size", "")
        if sz in ("1-10", "11-50", "51-200"):
            startup_score = 3.0
            break

    edu_score = 0.0
    relevant_fields = {"computer science", "machine learning", "artificial intelligence",
                       "ai", "data science", "mathematics", "statistics",
                       "computer engineering", "information technology"}
    for edu in education:
        field = (edu.get("field_of_study", "") or "").lower()
        tier = edu.get("tier", "unknown")
        is_relevant = any(rf in field for rf in relevant_fields)
        if is_relevant and tier == "tier_1":
            edu_score = max(edu_score, 2.0)
        elif is_relevant and tier == "tier_2":
            edu_score = max(edu_score, 1.5)
        elif is_relevant and tier == "tier_3":
            edu_score = max(edu_score, 0.5)

    raw = yoe_score + product_score + title_score + startup_score + edu_score
    max_raw = 25.0

    evidence["yoe"] = (yoe_score, 8.0, f"{yoe:.1f}yr experience (sweet spot: 5-9)")
    evidence["product_pct"] = (product_score, 7.0, f"{product_company_pct:.0%} non-consulting career")
    evidence["title"] = (title_score, 5.0, f"relevant title progression")
    evidence["startup"] = (startup_score, 3.0, f"startup experience")

    return raw / max_raw, evidence


def score_behavioral_fit(candidate: dict) -> tuple[float, dict]:
    signals = candidate.get("redrob_signals", {})
    evidence = {}

    try:
        from datetime import datetime
        last_active = signals.get("last_active_date", "")
        if last_active:
            days_inactive = (datetime(2026, 6, 1) - datetime.fromisoformat(last_active)).days
        else:
            days_inactive = 999
    except Exception:
        days_inactive = 999

    if days_inactive < 30:
        active_score = 5.0
    elif days_inactive < 90:
        active_score = 4.0
    elif days_inactive < 180:
        active_score = 3.0
    elif days_inactive < 365:
        active_score = 1.0
    else:
        active_score = 0.0

    response_rate = signals.get("recruiter_response_rate", 0)
    response_score = min(response_rate * 4.0, 4.0)

    open_to_work = signals.get("open_to_work_flag", False)
    if open_to_work and days_inactive <= 90:
        otw_score = 2.0
    elif open_to_work and days_inactive <= 180:
        otw_score = 1.0
    elif not open_to_work and days_inactive <= 30:
        otw_score = 0.5
    else:
        otw_score = 0.0

    notice = signals.get("notice_period_days", 90)
    if notice <= 30:
        notice_score = 4.0
    elif notice <= 60:
        notice_score = 2.5
    elif notice <= 90:
        notice_score = 1.5
    elif notice <= 120:
        notice_score = 0.5
    else:
        notice_score = 0.0

    interview_rate = signals.get("interview_completion_rate", 0)
    interview_score = min(interview_rate * 3.0, 3.0)

    avg_resp = signals.get("avg_response_time_hours", 168)
    if avg_resp < 24:
        resp_time_score = 2.0
    elif avg_resp < 48:
        resp_time_score = 1.5
    elif avg_resp < 72:
        resp_time_score = 1.0
    elif avg_resp < 168:
        resp_time_score = 0.5
    else:
        resp_time_score = 0.0

    completeness = signals.get("profile_completeness_score", 0)
    completeness_score = min(completeness / 100.0 * 2.0, 2.0)

    verified = 0.0
    if signals.get("verified_email", False):
        verified += 0.5
    if signals.get("verified_phone", False):
        verified += 0.5
    if signals.get("linkedin_connected", False):
        verified += 0.5

    offer_rate = signals.get("offer_acceptance_rate", -1)
    if offer_rate == -1:
        offer_score = 0.75
    else:
        offer_score = min(offer_rate * 1.5, 1.5)

    raw = active_score + response_score + otw_score + notice_score + interview_score + resp_time_score + completeness_score + verified + offer_score
    max_raw = 25.0

    evidence["active"] = (active_score, 5.0, f"last active {days_inactive}d ago")
    evidence["response"] = (response_score, 4.0, f"response rate {response_rate:.0%}")
    evidence["notice"] = (notice_score, 4.0, f"notice period {notice}d")

    return raw / max_raw, evidence


def score_location_fit(candidate: dict) -> tuple[float, dict]:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    evidence = {}

    country = (profile.get("country", "") or "").strip()
    location = normalize_city(profile.get("location", ""))

    india_score = 4.0 if country == "India" else 0.0

    city_score = 0.0
    if any(city in location for city in PREFERRED_CITIES):
        city_score = 3.0
    elif any(city in location for city in TIER1_INDIAN_CITIES):
        city_score = 2.0

    relocate = signals.get("willing_to_relocate", False)
    relocate_score = 1.5 if relocate and city_score < 3.0 else 0.0

    work_mode = (signals.get("preferred_work_mode", "") or "").lower()
    if work_mode in ("hybrid", "flexible"):
        mode_score = 1.5
    elif work_mode == "onsite":
        mode_score = 1.0
    else:
        mode_score = 0.0

    raw = india_score + city_score + relocate_score + mode_score
    max_raw = 10.0

    if country != "India":
        raw = min(raw, 2.0)

    evidence["india"] = (india_score, 4.0, f"country={country}")
    evidence["city"] = (city_score, 3.0, f"location={location}")

    return raw / max_raw, evidence


def compute_red_flags(candidate: dict) -> tuple[float, list[str]]:
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})
    flags = []

    career_text = " ".join(h.get("description", "") for h in career).lower()
    summary_text = (profile.get("summary", "") or "").lower()
    all_text = career_text + " " + summary_text
    has_positive = any(p in all_text for p in POSITIVE_CAREER_PHRASES)

    ai_skill_count = sum(1 for s in skills if skill_matches_taxonomy(s["name"], ALL_RELEVANT_SKILLS))

    if ai_skill_count >= 8 and not has_positive:
        non_tech = any(kw in all_text for kw in
            ["accounting", "marketing", "hr", "human resources", "content writing",
             "graphic design", "sales executive", "customer support", "operations manager"])
        if non_tech:
            flags.append(("keyword_stuffer", 0.15))

    if ai_skill_count >= 5 and not has_positive:
        flags.append(("skill_title_mismatch", 0.12))

    all_consulting = all(
        h.get("industry", "") in CONSULTING_INDUSTRIES for h in career
    ) if career else False
    if all_consulting and len(career) > 1:
        flags.append(("pure_consulting", 0.10))

    wrong_domain_count = sum(
        1 for s in skills if skill_matches_taxonomy(s["name"], WRONG_DOMAIN_SKILLS)
    )
    core_count = sum(
        1 for s in skills
        if skill_matches_taxonomy(s["name"], CORE_ML_SKILLS | RETRIEVAL_SKILLS | EMBEDDING_SKILLS | VECTOR_DB_SKILLS)
    )
    if wrong_domain_count > core_count and wrong_domain_count >= 3:
        flags.append(("wrong_domain", 0.08))

    if len(career) >= 3:
        avg_tenure = sum(h.get("duration_months", 0) for h in career) / len(career)
        if avg_tenure < 18:
            flags.append(("title_hopper", 0.05))

    salary_max = signals.get("expected_salary_range_inr_lpa", {}).get("max", 0)
    if salary_max > 45:
        flags.append(("excessive_salary", 0.03))

    total_penalty = sum(p for _, p in flags)
    return total_penalty, [f[0] for f in flags]


def combine_scores(candidate: dict) -> tuple[float, dict]:
    tech_score, tech_evidence = score_technical_fit(candidate)
    exp_score, exp_evidence = score_experience_fit(candidate)
    behav_score, behav_evidence = score_behavioral_fit(candidate)
    loc_score, loc_evidence = score_location_fit(candidate)
    penalty, flag_names = compute_red_flags(candidate)

    raw = (
        WEIGHTS["technical_fit"] * tech_score +
        WEIGHTS["experience_fit"] * exp_score +
        WEIGHTS["behavioral"] * behav_score +
        WEIGHTS["location"] * loc_score
    )

    # Minimum technical threshold: candidates with near-zero technical fit
    # should not rank high regardless of other dimensions
    if tech_score < 0.05:
        raw *= 0.4  # hard penalty for zero-tech candidates

    raw = max(raw - penalty, 0.0)

    all_evidence = {}
    all_evidence.update({f"tech_{k}": v for k, v in tech_evidence.items()})
    all_evidence.update({f"exp_{k}": v for k, v in exp_evidence.items()})
    all_evidence.update({f"behav_{k}": v for k, v in behav_evidence.items()})
    all_evidence.update({f"loc_{k}": v for k, v in loc_evidence.items()})
    if flag_names:
        all_evidence["red_flags"] = (penalty, penalty, f"flags: {', '.join(flag_names)}")

    return raw, all_evidence


def tier_normalize(raw: float, scores_list: list[float] | None = None) -> float:
    if scores_list and len(scores_list) > 10:
        percentiles = [0, 0.25, 0.50, 0.75, 1.0]
        sorted_scores = sorted(scores_list)
        n = len(sorted_scores)
        boundaries = [float(sorted_scores[min(int(p * n), n - 1)]) for p in [0, 0.25, 0.50, 0.75, 1.0]]
        if boundaries[-1] == boundaries[0]:
            return raw
        tiers = [
            (boundaries[3], boundaries[4], 0.85, 1.00),
            (boundaries[2], boundaries[3], 0.55, 0.85),
            (boundaries[1], boundaries[2], 0.25, 0.55),
            (boundaries[0], boundaries[1], 0.00, 0.25),
        ]
    else:
        tiers = SCORE_TIERS

    for lo, hi, out_lo, out_hi in tiers:
        if lo <= raw <= hi:
            if hi == lo:
                return (out_lo + out_hi) / 2
            return out_lo + (raw - lo) / (hi - lo) * (out_hi - out_lo)
    return 0.0