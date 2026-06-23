"""
scoring.py
----------
Combines the per-candidate feature dict (features.py) into a single
composite score, after applying the honeypot hard-rule (honeypot.py).

Design: an explicit, additive weighted-sum of explainable sub-scores
(title, skills, narrative, experience, company, location, education),
multiplied by a behavioral modifier and a job-hopper / honeypot penalty.
This is intentionally NOT a learned/black-box model: every weight is
defensible at the Stage 5 interview, the whole thing runs over precomputed
per-row features in well under a second for 100K rows (no GPU, no network,
trivially fits the 5 min / 16 GB CPU-only budget), and the score
decomposition is exactly what reasoning.py uses to write honest, specific
per-candidate justifications instead of templated praise.
"""

from datetime import date

from . import features as feat
from .honeypot import detect_honeypot, HONEYPOT_SCORE_MULT

WEIGHTS = {
    "title_score": 0.28,
    "skill_score": 0.24,
    "narrative_score": 0.12,
    "experience_score": 0.10,
    "company_score": 0.08,
    "location_score": 0.10,
    "education_score": 0.04,
}

JOB_HOPPER_PENALTY = 0.88


def score_candidate(candidate: dict, snapshot_date: date) -> dict:
    """Returns a flat dict: every feature field + 'raw_score' + 'is_honeypot'."""

    info = {}
    info.update(feat.title_fit(candidate))
    info.update(feat.skill_fit(candidate))
    info.update(feat.narrative_fit(candidate))
    info.update(feat.experience_fit(candidate))
    info.update(feat.company_fit(candidate))
    info.update(feat.location_fit(candidate))
    info.update(feat.education_fit(candidate))
    info.update(feat.behavioral_modifier(candidate, snapshot_date))

    base = sum(WEIGHTS[k] * info[k] for k in WEIGHTS)

    modifier = info["behavioral_modifier"]
    composite = base * modifier

    if info.get("job_hopper_flag"):
        composite *= JOB_HOPPER_PENALTY

    is_honeypot, honeypot_reason = detect_honeypot(candidate)
    if is_honeypot:
        composite *= HONEYPOT_SCORE_MULT

    info["base_score"] = base
    info["raw_score"] = composite
    info["is_honeypot"] = is_honeypot
    info["honeypot_reason"] = honeypot_reason
    return info
