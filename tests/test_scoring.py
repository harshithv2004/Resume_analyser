"""
tests/test_scoring.py
----------------------
Lightweight sanity tests, no external test framework required beyond
pytest. Run with:  pytest -q
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.honeypot import detect_honeypot
from src.scoring import score_candidate
from src.features import title_fit, experience_fit, location_fit


def _base_candidate(**overrides):
    c = {
        "candidate_id": "CAND_0000000",
        "profile": {
            "anonymized_name": "Test Candidate",
            "headline": "AI Engineer",
            "summary": "Built and shipped a production ranking system at scale.",
            "location": "Pune, Maharashtra",
            "country": "India",
            "years_of_experience": 6.5,
            "current_title": "Senior AI Engineer",
            "current_company": "Razorpay",
            "current_company_size": "1001-5000",
            "current_industry": "Fintech",
        },
        "career_history": [
            {
                "company": "Razorpay", "title": "Senior AI Engineer",
                "start_date": "2023-01-01", "end_date": None,
                "duration_months": 36, "is_current": True,
                "industry": "Fintech", "company_size": "1001-5000",
                "description": "Owned the ranking layer end to end.",
            },
            {
                "company": "Zoho", "title": "ML Engineer",
                "start_date": "2018-01-01", "end_date": "2022-12-01",
                "duration_months": 42, "is_current": False,
                "industry": "SaaS", "company_size": "5001-10000",
                "description": "Built recommendation models.",
            },
        ],
        "education": [
            {"institution": "IIT Bombay", "degree": "B.Tech",
             "field_of_study": "CS", "start_year": 2014, "end_year": 2018,
             "grade": "8.5", "tier": "tier_1"},
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 20, "duration_months": 60},
            {"name": "PyTorch", "proficiency": "advanced", "endorsements": 10, "duration_months": 40},
        ],
        "certifications": [],
        "languages": [],
        "redrob_signals": {
            "profile_completeness_score": 90,
            "signup_date": "2025-01-01",
            "last_active_date": "2026-05-20",
            "open_to_work_flag": True,
            "profile_views_received_30d": 50,
            "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.7,
            "avg_response_time_hours": 12,
            "skill_assessment_scores": {},
            "connection_count": 100,
            "endorsements_received": 30,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 30, "max": 45},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 50,
            "search_appearance_30d": 100,
            "saved_by_recruiters_30d": 5,
            "interview_completion_rate": 0.8,
            "offer_acceptance_rate": 0.6,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }
    for k, v in overrides.items():
        c[k] = v
    return c


def test_strong_candidate_scores_highly():
    c = _base_candidate()
    info = score_candidate(c, date(2026, 5, 27))
    assert info["raw_score"] > 0.8
    assert not info["is_honeypot"]


def test_honeypot_expert_zero_duration_detected():
    c = _base_candidate()
    c["skills"] = [
        {"name": "Python", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
    ]
    is_hp, reason = detect_honeypot(c)
    assert is_hp
    assert "0 months" in reason


def test_honeypot_yoe_career_history_mismatch_detected():
    c = _base_candidate()
    c["profile"]["years_of_experience"] = 15.0  # but career_history sums to ~6.5 yrs
    is_hp, reason = detect_honeypot(c)
    assert is_hp


def test_clean_candidate_not_flagged_honeypot():
    c = _base_candidate()
    is_hp, _ = detect_honeypot(c)
    assert not is_hp


def test_title_tier_exact_match_scores_max():
    c = _base_candidate()
    result = title_fit(c)
    assert result["current_title_tier"] == 5
    assert result["title_score"] > 0.9


def test_offdomain_title_scores_low():
    c = _base_candidate()
    c["profile"]["current_title"] = "Business Analyst"
    c["career_history"][0]["title"] = "Business Analyst"
    result = title_fit(c)
    assert result["current_title_tier"] == 0
    assert result["title_score"] < 0.3


def test_experience_fit_ideal_band():
    c = _base_candidate()
    c["profile"]["years_of_experience"] = 7.0
    assert experience_fit(c)["experience_score"] == 1.0


def test_experience_fit_far_outside_band_is_lower():
    c = _base_candidate()
    c["profile"]["years_of_experience"] = 25.0
    score = experience_fit(c)["experience_score"]
    assert 0.0 <= score < 1.0


def test_location_fit_india_pune_is_max():
    c = _base_candidate()
    result = location_fit(c)
    assert result["location_score"] == 1.0


def test_location_fit_outside_india_no_relocate_is_low():
    c = _base_candidate()
    c["profile"]["country"] = "USA"
    c["profile"]["location"] = "New York"
    c["redrob_signals"]["willing_to_relocate"] = False
    result = location_fit(c)
    assert result["location_score"] < 0.3


if __name__ == "__main__":
    import inspect
    tests = [f for name, f in list(globals().items()) if name.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"PASS: {t.__name__}")
    print(f"\n{passed}/{len(tests)} tests passed.")
