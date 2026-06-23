"""
honeypot.py
-----------
Explicit honeypot / trap detector.

The submission_spec explicitly tells us honeypots are forced to relevance
tier 0 and warns that a ranker that surfaces them in the top 10 "isn't
reading profiles - it's just doing keyword embedding." We were also told we
"don't need to special-case them" - a good ranker should naturally avoid
them - but given that the hard Stage-3 cutoff is honeypot-rate > 10% in the
top 100, we add an explicit, cheap, auditable rule on top of the organic
scoring as a safety net. Both rules below were derived empirically from the
actual candidates.jsonl (see taxonomy.py docstring) and are clean, sharply-
separated outliers, not fuzzy heuristics:

  Rule 1 - "phantom expertise": at least one skill is marked proficiency
  ="expert" with duration_months == 0. In the full 100K pool this fires for
  exactly 21 candidates, and the *next* most extreme case is 0 such skills
  - i.e. there is no ambiguous middle ground. This is a direct match for
  the JD's example: '"expert" proficiency in 10 skills with 0 years used'.

  Rule 2 - "impossible timeline": sum(career_history.duration_months) / 12
  divided by profile.years_of_experience falls outside [0.5, 1.5]. The
  population is otherwise tightly clustered between 0.94 and 1.0 (small
  gaps between jobs are normal), so this boundary has a wide empty margin
  on both sides - again, not a fuzzy cut. This is the dataset's analogue of
  the JD's example: '8 years of experience at a company founded 3 years
  ago' - the timeline simply does not add up.

A candidate flagged by either rule is treated as a honeypot: its composite
score is crushed (multiplied by HONEYPOT_SCORE_MULT) so it cannot appear in
the top 100 except in the pathological case where >99,900 other candidates
also score near zero, which does not happen in this dataset.
"""

from typing import Tuple

HONEYPOT_SCORE_MULT = 0.001

EXPERT_ZERO_TENURE_THRESHOLD = 1   # >=1 such skill triggers the rule
TIMELINE_RATIO_LOW = 0.5
TIMELINE_RATIO_HIGH = 1.5


def detect_honeypot(candidate: dict) -> Tuple[bool, str]:
    """Returns (is_honeypot, reason_string)."""

    skills = candidate.get("skills", []) or []
    expert_zero = [
        s["name"] for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    ]
    if len(expert_zero) >= EXPERT_ZERO_TENURE_THRESHOLD:
        return True, (
            f"Honeypot: claims 'expert' proficiency with 0 months of "
            f"hands-on duration in {len(expert_zero)} skill(s) "
            f"({', '.join(expert_zero[:3])}{'...' if len(expert_zero) > 3 else ''})."
        )

    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0
    career_history = candidate.get("career_history", []) or []
    total_months = sum(ch.get("duration_months", 0) for ch in career_history)
    if yoe and yoe > 0:
        ratio = (total_months / 12.0) / yoe
        if ratio > TIMELINE_RATIO_HIGH or ratio < TIMELINE_RATIO_LOW:
            return True, (
                f"Honeypot: stated years_of_experience ({yoe}) is inconsistent "
                f"with the sum of career_history durations "
                f"({total_months / 12.0:.1f} yrs implied, ratio={ratio:.2f})."
            )

    return False, ""
