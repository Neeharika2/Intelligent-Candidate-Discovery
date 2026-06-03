from datetime import datetime
from typing import Any, Dict, List

_TITLE_DOMAINS = {
    "ml": {"ml", "ai", "machine learning", "nlp", "search", "ranking"},
    "marketing": {"marketing", "seo", "growth", "brand", "content"},
    "sales": {"sales", "account", "business development"},
    "ops": {"operations", "support", "hr", "finance", "admin"},
}


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _domain_from_text(text: str) -> str:
    lowered = text.lower()
    for domain, keywords in _TITLE_DOMAINS.items():
        for keyword in keywords:
            if keyword in lowered:
                return domain
    return "other"


def _count_overlap_months(roles: List[Dict[str, Any]]) -> int:
    ranges = []
    for role in roles:
        start = role.get("start_date")
        if not start:
            continue
        end = role.get("end_date")
        try:
            start_dt = _parse_date(start)
        except ValueError:
            continue
        if end:
            try:
                end_dt = _parse_date(end)
            except ValueError:
                end_dt = start_dt
        else:
            end_dt = datetime.today()
        ranges.append((start_dt, end_dt))

    overlap_months = 0
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            start = max(ranges[i][0], ranges[j][0])
            end = min(ranges[i][1], ranges[j][1])
            if end > start:
                overlap_months += max(0, int((end - start).days / 30))
    return overlap_months


def detect_honeypots(candidates: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    flags: Dict[str, Dict[str, Any]] = {}

    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        profile = candidate.get("profile", {})
        skills = candidate.get("skills", [])
        career = candidate.get("career_history", [])
        education = candidate.get("education", [])

        reasons: List[str] = []

        expert_zero = sum(
            1
            for skill in skills
            if skill.get("proficiency") == "expert"
            and int(skill.get("duration_months", 1) or 0) == 0
        )
        if expert_zero >= 2:
            reasons.append("expert_skills_with_zero_duration")

        advanced_plus = sum(
            1
            for skill in skills
            if skill.get("proficiency") in {"advanced", "expert"}
        )
        yoe = profile.get("years_of_experience", 0) or 0
        if advanced_plus > 8 and yoe < 5:
            reasons.append("too_many_advanced_skills_for_experience")

        total_months = sum(role.get("duration_months", 0) or 0 for role in career)
        if abs((yoe * 12) - total_months) > 48:
            reasons.append("career_months_mismatch")

        overlap_months = _count_overlap_months(career)
        if overlap_months > 6:
            reasons.append("overlapping_roles")

        title_domain = _domain_from_text(profile.get("current_title", ""))
        description_text = " ".join(role.get("description", "") for role in career)
        desc_domain = _domain_from_text(description_text)
        if title_domain != "other" and desc_domain != "other" and title_domain != desc_domain:
            reasons.append("title_description_domain_mismatch")

        if education and career:
            end_years = [edu.get("end_year") for edu in education if edu.get("end_year")]
            if end_years:
                latest_edu = max(end_years)
                role_starts = []
                for role in career:
                    start = role.get("start_date")
                    if start:
                        try:
                            role_starts.append(_parse_date(start).year)
                        except ValueError:
                            pass
                if role_starts and min(role_starts) + 1 < latest_edu:
                    reasons.append("career_before_education_completion")

        if len(reasons) >= 3:
            flags[candidate_id] = {
                "red_flags": len(reasons),
                "reasons": reasons,
            }

    return flags
