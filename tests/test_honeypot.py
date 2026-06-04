import json
import pytest
from src.honeypot import detect_honeypot


def make_candidate(**overrides):
    base = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Test User",
            "headline": "Engineer",
            "summary": "Test summary",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 6.0,
            "current_title": "ML Engineer",
            "current_company": "TestCo",
            "current_company_size": "51-200",
            "current_industry": "Software",
        },
        "career_history": [
            {
                "company": "TestCo",
                "title": "ML Engineer",
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 72,
                "is_current": True,
                "industry": "Software",
                "company_size": "51-200",
                "description": "Built ML systems",
            }
        ],
        "education": [
            {
                "institution": "IIT",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2014,
                "end_year": 2018,
                "grade": "9.0",
                "tier": "tier_1",
            }
        ],
        "skills": [
            {"name": "Python", "proficiency": "advanced", "endorsements": 20, "duration_months": 60}
        ],
        "redrob_signals": {
            "profile_completeness_score": 80,
            "signup_date": "2024-01-01",
            "last_active_date": "2026-05-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 10,
            "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.5,
            "avg_response_time_hours": 24,
            "skill_assessment_scores": {},
            "connection_count": 100,
            "endorsements_received": 20,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 20, "max": 35},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 50,
            "search_appearance_30d": 50,
            "saved_by_recruiters_30d": 5,
            "interview_completion_rate": 0.8,
            "offer_acceptance_rate": 0.7,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }
    for key, value in overrides.items():
        if key == "years_of_experience":
            base["profile"]["years_of_experience"] = value
        elif key == "career_history":
            base["career_history"] = value
        elif key == "skills":
            base["skills"] = value
        elif key == "education":
            base["education"] = value
        elif key == "skill_assessment_scores":
            base["redrob_signals"]["skill_assessment_scores"] = value
    return base


def test_normal_candidate_not_flagged():
    c = make_candidate()
    flagged, reason = detect_honeypot(c)
    assert not flagged


def test_impossible_career_timeline():
    c = make_candidate(years_of_experience=2)
    c["career_history"] = [
        {**c["career_history"][0], "duration_months": 80},  # 80 > 2*12*1.8+24 = 67.2
    ]
    flagged, reason = detect_honeypot(c)
    assert flagged
    assert "career_months" in reason


def test_expert_skills_zero_duration():
    c = make_candidate()
    c["skills"] = [
        {"name": "Python", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
        {"name": "ML", "proficiency": "expert", "endorsements": 3, "duration_months": 0},
        {"name": "RAG", "proficiency": "expert", "endorsements": 1, "duration_months": 0},
    ]
    flagged, reason = detect_honeypot(c)
    assert flagged
    assert "expert skills with zero duration" in reason


def test_impossible_education_duration():
    c = make_candidate()
    c["education"] = [
        {
            "institution": "Diploma Mill",
            "degree": "Ph.D",
            "field_of_study": "AI",
            "start_year": 2010,
            "end_year": 2025,
            "grade": "A",
            "tier": "tier_4",
        }
    ]
    flagged, reason = detect_honeypot(c)
    assert flagged
    assert "education duration" in reason


def test_assessment_contradicts_proficiency():
    c = make_candidate()
    c["skills"] = [
        {"name": "NLP", "proficiency": "expert", "endorsements": 30, "duration_months": 36},
    ]
    c["redrob_signals"]["skill_assessment_scores"] = {"NLP": 15}
    flagged, reason = detect_honeypot(c)
    assert flagged
    assert "expert claim contradicted" in reason


def test_current_role_zero_months():
    c = make_candidate()
    c["career_history"][0]["duration_months"] = 0
    c["career_history"][0]["is_current"] = True
    flagged, reason = detect_honeypot(c)
    assert flagged
    assert "current role with 0 months" in reason


def test_expert_skills_no_endorsements():
    c = make_candidate()
    c["skills"] = [
        {"name": "A", "proficiency": "expert", "endorsements": 0, "duration_months": 12},
        {"name": "B", "proficiency": "expert", "endorsements": 0, "duration_months": 12},
        {"name": "C", "proficiency": "expert", "endorsements": 0, "duration_months": 12},
        {"name": "D", "proficiency": "expert", "endorsements": 0, "duration_months": 12},
        {"name": "E", "proficiency": "expert", "endorsements": 0, "duration_months": 12},
    ]
    flagged, reason = detect_honeypot(c)
    assert flagged
    assert "expert skills" in reason and "0 endorsements" in reason