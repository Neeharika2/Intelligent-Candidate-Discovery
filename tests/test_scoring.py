import json
from src.scoring import (
    score_technical_fit, score_experience_fit, score_behavioral_fit,
    score_location_fit, compute_red_flags, combine_scores, tier_normalize,
)


def load_sample_candidates():
    with open("sample/sample_candidates.json", "r", encoding="utf-8") as f:
        return json.load(f)


def test_scoring_runs_on_all_samples():
    candidates = load_sample_candidates()
    for c in candidates:
        raw, evidence = combine_scores(c)
        assert 0.0 <= raw <= 1.0, f"Score out of range for {c['candidate_id']}: {raw}"


def test_technical_fit_ml_engineer_higher_than_hr():
    candidates = load_sample_candidates()
    scores = {}
    for c in candidates:
        tech_score, _ = score_technical_fit(c)
        title = (c.get("profile", {}).get("current_title", "") or "").lower()
        scores[c["candidate_id"]] = (tech_score, title)
    ml_scored = [(cid, s) for cid, (s, t) in scores.items() if "engineer" in t or "data" in t or "ml" in t]
    non_tech = [(cid, s) for cid, (s, t) in scores.items() if "manager" in t or "accountant" in t or "support" in t]
    if ml_scored and non_tech:
        avg_ml = sum(s for _, s in ml_scored) / len(ml_scored)
        avg_non = sum(s for _, s in non_tech) / len(non_tech)
        assert avg_ml > avg_non, "ML engineers should score higher than managers on technical fit"


def test_location_india_preferred():
    candidates = load_sample_candidates()
    india_scores = []
    non_india_scores = []
    for c in candidates:
        loc_score, _ = score_location_fit(c)
        if c.get("profile", {}).get("country") == "India":
            india_scores.append(loc_score)
        else:
            non_india_scores.append(loc_score)
    if india_scores and non_india_scores:
        assert max(india_scores) >= max(non_india_scores), "India-based candidates should score higher on location"


def test_tier_normalize_produces_valid_range():
    scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    for s in scores:
        result = tier_normalize(s)
        assert 0.0 <= result <= 1.0, f"Tier normalize produced out-of-range: {result}"


def test_red_flags_pure_consulting():
    c = {
        "candidate_id": "CAND_TEST",
        "profile": {"current_title": "Operations Manager", "current_industry": "IT Services",
                     "years_of_experience": 10, "summary": "I manage teams", "headline": "Ops",
                     "location": "Mumbai", "country": "India", "current_company": "Wipro", "current_company_size": "10001+"},
        "career_history": [
            {"company": "Infosys", "title": "Analyst", "start_date": "2014-01-01", "end_date": None,
             "duration_months": 60, "is_current": True, "industry": "IT Services", "company_size": "10001+", "description": "Consulting"},
            {"company": "TCS", "title": "Associate", "start_date": "2010-01-01", "end_date": "2014-01-01",
             "duration_months": 48, "is_current": False, "industry": "IT Services", "company_size": "10001+", "description": "More consulting"},
        ],
        "education": [],
        "skills": [],
        "redrob_signals": {"notice_period_days": 90, "recruiter_response_rate": 0.3,
                          "last_active_date": "2026-01-01", "open_to_work_flag": False,
                          "github_activity_score": -1, "offer_acceptance_rate": -1,
                          "interview_completion_rate": 0.5, "avg_response_time_hours": 72,
                          "profile_completeness_score": 50, "verified_email": True,
                          "verified_phone": False, "linkedin_connected": False,
                          "expected_salary_range_inr_lpa": {"min": 10, "max": 20},
                          "preferred_work_mode": "hybrid", "willing_to_relocate": False,
                          "profile_views_received_30d": 10, "applications_submitted_30d": 2,
                          "connection_count": 100, "endorsements_received": 5,
                          "saved_by_recruiters_30d": 1, "search_appearance_30d": 50,
                          "skill_assessment_scores": {}, "signup_date": "2024-01-01"},
    }
    penalty, flags = compute_red_flags(c)
    assert "pure_consulting" in flags, f"Should flag pure consulting career, got {flags}"


def test_behavioral_sentinel_values():
    c = {
        "candidate_id": "CAND_TEST",
        "profile": {},
        "career_history": [],
        "education": [],
        "skills": [],
        "redrob_signals": {
            "github_activity_score": -1,
            "offer_acceptance_rate": -1,
            "recruiter_response_rate": 0.5,
            "last_active_date": "2026-05-01",
            "open_to_work_flag": True,
            "notice_period_days": 30,
            "interview_completion_rate": 0.8,
            "avg_response_time_hours": 24,
            "profile_completeness_score": 80,
            "verified_email": True, "verified_phone": True, "linkedin_connected": True,
            "expected_salary_range_inr_lpa": {"min": 15, "max": 30},
            "preferred_work_mode": "hybrid", "willing_to_relocate": True,
            "profile_views_received_30d": 20, "applications_submitted_30d": 3,
            "connection_count": 200, "endorsements_received": 10,
            "saved_by_recruiters_30d": 3, "search_appearance_30d": 40,
            "skill_assessment_scores": {}, "signup_date": "2024-01-01",
        },
    }
    score, evidence = score_behavioral_fit(c)
    assert score > 0, f"Behavioral score should be positive: {score}"