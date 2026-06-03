from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.skill_taxonomy import normalize_skill


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_city(location: str) -> str:
    if not location:
        return ""
    return location.split(",")[0].strip().lower()


def _normalize_skills(skills: List[Dict[str, Any]]) -> List[str]:
    normalized = []
    for skill in skills:
        name = normalize_skill(skill.get("name", ""))
        if name:
            normalized.append(name)
    return normalized


def build_skills_text(skills: List[Dict[str, Any]]) -> str:
    normalized = _normalize_skills(skills)
    return ", ".join(sorted(set(normalized)))


def _chunk_text(text: str, chunk_size: int = 1200) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size
    return chunks


def build_profile_chunks(candidate: Dict[str, Any], max_chunks: int = 6) -> List[Tuple[str, float]]:
    profile = candidate.get("profile", {})
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    chunks: List[Tuple[str, float]] = []

    if headline:
        chunks.append((headline, 1.2))
    if summary:
        chunks.append((summary, 1.3))

    career = candidate.get("career_history", [])
    roles = []
    for role in career:
        start_date = role.get("start_date")
        duration = role.get("duration_months", 0) or 0
        if start_date:
            try:
                start = _parse_date(start_date)
            except ValueError:
                start = datetime.min
        else:
            start = datetime.min
        roles.append((start, duration, role))

    roles.sort(key=lambda item: item[0], reverse=True)

    for idx, (_, duration, role) in enumerate(roles):
        description = role.get("description", "")
        if not description:
            continue
        weight = 1.0 + min(duration / 24.0, 1.5)
        weight *= max(0.6, 1.0 - (idx * 0.1))
        for chunk in _chunk_text(description):
            chunks.append((chunk, weight))

    if not chunks:
        skills_text = build_skills_text(candidate.get("skills", []))
        fallback = " ".join([headline, summary, skills_text]).strip()
        if fallback:
            chunks.append((fallback, 1.0))

    return chunks[:max_chunks]


def build_profile_text(candidate: Dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    role_text = " ".join(
        role.get("description", "")
        for role in candidate.get("career_history", [])
        if role.get("description")
    )
    return " ".join([headline, summary, role_text]).strip()


def _career_months(career_history: List[Dict[str, Any]]) -> int:
    return int(sum(role.get("duration_months", 0) or 0 for role in career_history))


def _avg_tenure(career_history: List[Dict[str, Any]]) -> float:
    durations = [role.get("duration_months", 0) or 0 for role in career_history]
    if not durations:
        return 0.0
    return float(np.mean(durations))


def _gap_months(career_history: List[Dict[str, Any]]) -> int:
    roles = []
    for role in career_history:
        start = role.get("start_date")
        end = role.get("end_date")
        if not start:
            continue
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
        roles.append((start_dt, end_dt))

    if len(roles) < 2:
        return 0

    roles.sort(key=lambda item: item[0])
    gaps = 0
    for (prev_start, prev_end), (next_start, _) in zip(roles, roles[1:]):
        gap_days = (next_start - prev_end).days
        if gap_days > 30:
            gaps += int(gap_days / 30)
    return gaps


def build_feature_frame(candidates: List[Dict[str, Any]]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for candidate in candidates:
        profile = candidate.get("profile", {})
        signals = candidate.get("redrob_signals", {})
        career = candidate.get("career_history", [])
        skills = candidate.get("skills", [])

        record = {
            "candidate_id": candidate.get("candidate_id"),
            "years_of_experience": _safe_float(profile.get("years_of_experience")),
            "total_career_months": _career_months(career),
            "avg_tenure_months": _avg_tenure(career),
            "gap_months": _gap_months(career),
            "current_title": profile.get("current_title", ""),
            "current_company": profile.get("current_company", ""),
            "current_industry": profile.get("current_industry", ""),
            "location": profile.get("location", ""),
            "location_city": _extract_city(profile.get("location", "")),
            "country": profile.get("country", ""),
            "profile_text": build_profile_text(candidate),
            "skills_text": build_skills_text(skills),
            "recruiter_response_rate": _safe_float(signals.get("recruiter_response_rate")),
            "avg_response_time_hours": _safe_float(signals.get("avg_response_time_hours")),
            "last_active_date": signals.get("last_active_date"),
            "open_to_work_flag": bool(signals.get("open_to_work_flag")),
            "notice_period_days": int(signals.get("notice_period_days", 0) or 0),
            "preferred_work_mode": signals.get("preferred_work_mode", ""),
            "willing_to_relocate": bool(signals.get("willing_to_relocate")),
            "profile_completeness_score": _safe_float(signals.get("profile_completeness_score")),
            "interview_completion_rate": _safe_float(signals.get("interview_completion_rate")),
            "verified_email": bool(signals.get("verified_email")),
            "verified_phone": bool(signals.get("verified_phone")),
        }
        records.append(record)

    return pd.DataFrame.from_records(records)
