#!/usr/bin/env python3
"""
scripts/eda.py
---------------
Reproduces every number cited in README.md / src/taxonomy.py docstrings,
so the methodology claims aren't just asserted - they're re-runnable
against your own copy of candidates.jsonl.

    python scripts/eda.py --candidates ./candidates.jsonl

Read-only, single streaming pass, stdlib only.
"""

import argparse
import collections
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    args = ap.parse_args()

    titles = collections.Counter()
    skills = collections.Counter()
    countries = collections.Counter()
    companies = collections.Counter()
    industries = collections.Counter()

    honeypot_expert_zero = []
    honeypot_yoe_mismatch = []

    n = 0
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            n += 1

            titles[d["profile"]["current_title"]] += 1
            countries[d["profile"]["country"]] += 1
            companies[d["profile"]["current_company"]] += 1
            industries[d["profile"]["current_industry"]] += 1
            for s in d.get("skills", []):
                skills[s["name"]] += 1

            expert_zero = [
                s["name"] for s in d.get("skills", [])
                if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
            ]
            if expert_zero:
                honeypot_expert_zero.append(d["candidate_id"])

            yoe = d["profile"].get("years_of_experience", 0) or 0
            total_months = sum(ch.get("duration_months", 0) for ch in d.get("career_history", []))
            if yoe > 0:
                ratio = (total_months / 12.0) / yoe
                if ratio > 1.5 or ratio < 0.5:
                    honeypot_yoe_mismatch.append(d["candidate_id"])

    print(f"Total candidates scanned: {n}\n")

    print(f"Unique titles: {len(titles)}")
    for t, c in titles.most_common():
        print(f"  {c:6d}  {t}")

    print(f"\nUnique skills: {len(skills)}")
    print("  (top 10 by frequency)")
    for s, c in skills.most_common(10):
        print(f"  {c:6d}  {s}")
    print("  (bottom 10 by frequency - the 'rare plain-language' tier)")
    for s, c in skills.most_common()[-10:]:
        print(f"  {c:6d}  {s}")

    print(f"\nCountries: {dict(countries.most_common(10))}")
    print(f"\nTop companies: {dict(companies.most_common(15))}")
    print(f"\nIndustries: {dict(industries.most_common(15))}")

    overlap = set(honeypot_expert_zero) & set(honeypot_yoe_mismatch)
    print(f"\nHoneypot rule 1 (expert+0 duration) fired on: {len(honeypot_expert_zero)} candidates")
    print(f"Honeypot rule 2 (yoe/career-history ratio outlier) fired on: {len(honeypot_yoe_mismatch)} candidates")
    print(f"Overlap between the two rules: {len(overlap)} (should be ~0 - independent signals)")
    print(f"Union (total flagged honeypots): {len(set(honeypot_expert_zero) | set(honeypot_yoe_mismatch))}")


if __name__ == "__main__":
    main()
