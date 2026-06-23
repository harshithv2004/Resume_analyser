"""
features.py
------------
Pure feature-extraction functions. Each function takes a raw candidate
dict (as parsed from one line of candidates.jsonl) and returns a small,
JSON-serialisable dict of named sub-scores plus any "evidence" needed later
for reasoning generation. No I/O, no global state -> easy to unit test.

All sub-scores are designed to land roughly in [0, 1] before the weights in
scoring.py are applied.
"""

import math
from datetime import date, datetime
from typing import Optional

from . import taxonomy as tx


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# 1. Title fit (current title + recency-weighted career history)
# ---------------------------------------------------------------------------
def title_fit(candidate: dict) -> dict:
    profile = candidate.get("profile", {})
    current_title = profile.get("current_title", "")
    current_tier = tx.TITLE_TIER.get(current_title, tx.DEFAULT_TITLE_TIER)

    # Look at career history too (most recent first), recency-weighted, so a
    # recent pivot into/out of ML shows up but doesn't dominate over the
    # current role.
    history = candidate.get("career_history", []) or []

    def _start(ch):
        d = _parse_date(ch.get("start_date"))
        return d or date(1970, 1, 1)

    history_sorted = sorted(history, key=_start, reverse=True)
    hist_score = 0.0
    hist_weight_total = 0.0
    for i, ch in enumerate(history_sorted[:4]):  # most recent 4 roles
        w = 1.0 / (i + 1)  # 1, 0.5, 0.33, 0.25
        tier = tx.TITLE_TIER.get(ch.get("title", ""), tx.DEFAULT_TITLE_TIER)
        hist_score += w * tx.TITLE_TIER_SCORE[tier]
        hist_weight_total += w

    hist_avg = hist_score / hist_weight_total if hist_weight_total else 0.0
    current_score = tx.TITLE_TIER_SCORE[current_tier]

    # current title weighted more heavily than history trend
    combined = 0.7 * current_score + 0.3 * hist_avg

    return {
        "title_score": _clip(combined),
        "current_title": current_title,
        "current_title_tier": current_tier,
    }


# ---------------------------------------------------------------------------
# 2. Skill fit (tiered, tenure & proficiency trust-weighted)
# ---------------------------------------------------------------------------
def skill_fit(candidate: dict) -> dict:
    skills = candidate.get("skills", []) or []

    core_hits, buzz_hits, rare_hits = [], [], []
    raw = 0.0
    for s in skills:
        name = s.get("name", "")
        prof = s.get("proficiency", "intermediate")
        dur = s.get("duration_months", 0) or 0
        prof_mult = tx.PROFICIENCY_MULT.get(prof, 0.5)
        # tenure trust: ramps from ~0.2 at 0 months to 1.0 at ~18 months.
        # An "expert" tag with 0 months is additionally zeroed out here
        # (independent of the global honeypot rule) so a single suspicious
        # skill doesn't get full credit even on an otherwise-clean profile.
        trust = min(1.0, (dur + 3) / 18.0)
        if prof == "expert" and dur == 0:
            trust = 0.0

        if name in tx.CORE_SKILLS:
            w = tx.CORE_SKILL_WEIGHT
            core_hits.append(name)
        elif name in tx.BUZZWORD_SKILLS:
            w = tx.BUZZWORD_SKILL_WEIGHT
            buzz_hits.append(name)
        elif name in tx.RARE_SKILLS:
            w = tx.RARE_SKILL_WEIGHT
            rare_hits.append(name)
        else:
            w = 0.0  # generic, non-discriminative skill

        raw += w * prof_mult * trust

    # log-dampened normalisation: empirically calibrated so a candidate with
    # ~6-8 well-evidenced core/rare skills tops out near 1.0, while a
    # handful of trust-gated buzzwords on a thin profile stays low.
    norm_const = 6.0
    skill_score = _clip(math.log1p(raw) / math.log1p(norm_const))

    named_hits = [s.get("name") for s in skills if s.get("name") in tx.JD_NAMED_SKILLS]

    return {
        "skill_score": skill_score,
        "core_skills_matched": core_hits,
        "buzzword_skills_matched": buzz_hits,
        "rare_skills_matched": rare_hits,
        "jd_named_skills_matched": named_hits,
    }


# ---------------------------------------------------------------------------
# 3. Production-narrative text mining (career_history descriptions + summary)
# ---------------------------------------------------------------------------
def narrative_fit(candidate: dict) -> dict:
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", []) or []
    text = (profile.get("summary", "") + " " + " ".join(
        ch.get("description", "") for ch in history
    )).lower()

    hits = [p for p in tx.PRODUCTION_PHRASES if p in text]
    concern_hits = [p for p in tx.PURE_RESEARCH_PHRASES if p in text]

    # diminishing returns past ~6 distinct phrase hits
    narrative_score = _clip(math.log1p(len(hits)) / math.log1p(6))
    if concern_hits:
        narrative_score *= 0.5  # explicit "no production deployment" language

    return {
        "narrative_score": narrative_score,
        "narrative_phrases_matched": hits[:6],
        "research_only_concern": bool(concern_hits),
    }


# ---------------------------------------------------------------------------
# 4. Experience fit (ideal band 5-9 yrs, soft falloff)
# ---------------------------------------------------------------------------
def experience_fit(candidate: dict) -> dict:
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0
    if 5 <= yoe <= 9:
        score = 1.0
    elif yoe < 5:
        score = _clip(1.0 - (5 - yoe) * 0.16)
    else:
        score = _clip(1.0 - (yoe - 9) * 0.10)
    return {"experience_score": _clip(score), "years_of_experience": yoe}


# ---------------------------------------------------------------------------
# 5. Company / org fit
# ---------------------------------------------------------------------------
def company_fit(candidate: dict) -> dict:
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", []) or []
    current_company = profile.get("current_company", "")
    current_industry = profile.get("current_industry", "")

    companies = {ch.get("company") for ch in history} | {current_company}
    companies.discard(None)

    consulting_only = bool(companies) and companies.issubset(tx.CONSULTING_FIRMS)

    is_ai_native = current_industry in tx.AI_NATIVE_INDUSTRIES
    is_product_co = (current_company in tx.PRODUCT_COMPANIES) or any(
        c in tx.PRODUCT_COMPANIES for c in companies
    )

    score = 0.5  # neutral baseline (large generic employer)
    if consulting_only:
        score = 0.15
    elif is_ai_native:
        score = 1.0
    elif is_product_co:
        score = 0.78

    # job-hopping / "title-chaser" heuristic: many short stints
    n_jobs = len(history)
    total_months = sum(ch.get("duration_months", 0) for ch in history)
    avg_tenure = total_months / n_jobs if n_jobs else 999
    job_hopper = n_jobs >= 3 and avg_tenure < 16

    return {
        "company_score": score,
        "consulting_only_history": consulting_only,
        "is_ai_native_company": is_ai_native,
        "is_product_company": is_product_co,
        "job_hopper_flag": job_hopper,
        "avg_tenure_months": round(avg_tenure, 1) if n_jobs else None,
    }


# ---------------------------------------------------------------------------
# 6. Location fit
# ---------------------------------------------------------------------------
def location_fit(candidate: dict) -> dict:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    country = (profile.get("country") or "").strip().lower()
    location = (profile.get("location") or "").strip().lower()
    willing = bool(signals.get("willing_to_relocate", False))

    if country == "india":
        if any(c in location for c in tx.PRIMARY_CITIES):
            score = 1.0
        elif any(c in location for c in tx.WELCOME_CITIES):
            score = 0.9
        else:
            score = 0.75  # other India city - still hybrid-feasible, no visa issue
    else:
        # JD: "Outside India: case-by-case, but we don't sponsor work visas."
        score = 0.45 if willing else 0.15

    return {"location_score": score, "country": profile.get("country"),
            "location": profile.get("location"), "willing_to_relocate": willing}


# ---------------------------------------------------------------------------
# 7. Education fit (minor signal)
# ---------------------------------------------------------------------------
_EDU_TIER_SCORE = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.3, "unknown": 0.4}


def education_fit(candidate: dict) -> dict:
    edu = candidate.get("education", []) or []
    if not edu:
        return {"education_score": 0.4, "best_education_tier": None}
    best = max(edu, key=lambda e: _EDU_TIER_SCORE.get(e.get("tier", "unknown"), 0.4))
    tier = best.get("tier", "unknown")
    return {"education_score": _EDU_TIER_SCORE.get(tier, 0.4), "best_education_tier": tier,
             "best_institution": best.get("institution")}


# ---------------------------------------------------------------------------
# 8. Behavioral signal multiplier
# ---------------------------------------------------------------------------
def behavioral_modifier(candidate: dict, snapshot_date: date) -> dict:
    s = candidate.get("redrob_signals", {})

    open_to_work = bool(s.get("open_to_work_flag", False))
    last_active = _parse_date(s.get("last_active_date"))
    days_inactive = (snapshot_date - last_active).days if last_active else 9999

    # recency: 1.0 if active within 2 weeks, decays to ~0.55 by 6 months
    recency_score = _clip(1.0 - days_inactive / 365.0, 0.4, 1.0)

    resp_rate = _clip(s.get("recruiter_response_rate", 0.0) or 0.0)
    completeness = _clip((s.get("profile_completeness_score", 0) or 0) / 100.0)
    verified_bonus = (
        (0.03 if s.get("verified_email") else 0)
        + (0.03 if s.get("verified_phone") else 0)
        + (0.02 if s.get("linkedin_connected") else 0)
    )

    notice = s.get("notice_period_days", 90) or 90
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.8
    else:
        notice_score = _clip(1.0 - (notice - 60) / 240.0, 0.4, 0.8)

    interview_completion = _clip(s.get("interview_completion_rate", 0.5) or 0.5)

    demand_raw = (s.get("search_appearance_30d", 0) or 0) + 3 * (s.get("saved_by_recruiters_30d", 0) or 0)
    demand_score = _clip(math.log1p(demand_raw) / math.log1p(300))

    offer_accept = s.get("offer_acceptance_rate", -1)
    offer_term = 0.0 if offer_accept is None or offer_accept < 0 else (offer_accept - 0.5) * 0.06

    weighted = (
        0.30 * (1.0 if open_to_work else 0.55)
        + 0.20 * recency_score
        + 0.20 * resp_rate
        + 0.10 * completeness
        + 0.10 * notice_score
        + 0.05 * interview_completion
        + 0.05 * demand_score
    )
    modifier = _clip(0.55 + 0.65 * weighted + verified_bonus + offer_term, 0.40, 1.25)

    return {
        "behavioral_modifier": modifier,
        "open_to_work_flag": open_to_work,
        "days_since_active": days_inactive,
        "recruiter_response_rate": resp_rate,
        "notice_period_days": notice,
        "profile_completeness_score": s.get("profile_completeness_score"),
    }
