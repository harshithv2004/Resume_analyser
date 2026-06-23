#!/usr/bin/env python3
"""
rank.py
-------
Single-command entry point for the Redrob Hackathon submission.

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Design goals (all driven by submission_spec.docx Section 3):
  * CPU only, no GPU, no network calls anywhere in this file or src/.
  * Wall-clock budget 5 minutes / 16 GB RAM for the FULL 100K pool.
  * Deterministic - re-running produces byte-identical output.

Pipeline (3 lightweight streaming passes over the JSONL/JSONL.GZ file,
chosen over loading everything into memory at once to keep RAM flat
regardless of pool size):

  Pass 0 (regex, no JSON parsing): extract every `last_active_date` value
          to compute a single "snapshot date" (max date seen) used as
          "today" for all recency calculations. Doing this without a full
          JSON parse keeps it very cheap.
  Pass 1 (full parse, O(1) memory per row): compute the full composite
          score for every candidate. Only (raw_score, candidate_id) is
          retained per row - the parsed dict is discarded immediately
          after scoring. This is the only pass that has to touch all
          100K rows with full feature computation, and it is what's
          checked against the 5 min / 16 GB constraint.
  Pass 2 (full parse, filtered): re-read the file, and for the ~100
          candidate_ids selected in Pass 1, recompute features (cheap)
          and generate the reasoning string. This avoids holding 100K
          parsed candidate dicts in memory just to keep 100 of them.

Output: a CSV at --out matching submission_spec.docx Section 2 exactly:
  header `candidate_id,rank,score,reasoning`, 100 data rows, rank 1..100
  each used once, score non-increasing with rank, ties broken by
  candidate_id ascending.
"""

import argparse
import csv
import gzip
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.scoring import score_candidate
from src.reasoning import generate_reasoning

TOP_N = 100
LAST_ACTIVE_RE = re.compile(r'"last_active_date"\s*:\s*"(\d{4}-\d{2}-\d{2})"')


def _opener(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, "r", encoding="utf-8")


def compute_snapshot_date(path: str) -> date:
    """Pass 0: cheap regex scan for the max last_active_date in the pool."""
    max_date = date(2000, 1, 1)
    with _opener(path) as f:
        for line in f:
            m = LAST_ACTIVE_RE.search(line)
            if m:
                y, mo, d = (int(x) for x in m.group(1).split("-"))
                cand = date(y, mo, d)
                if cand > max_date:
                    max_date = cand
    return max_date


def pass1_score_all(path: str, snapshot_date: date):
    """Returns list of (raw_score, candidate_id) for every row, O(1) extra
    memory per row (nothing is retained after scoring)."""
    results = []
    n = 0
    with _opener(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidate = json.loads(line)
            info = score_candidate(candidate, snapshot_date)
            results.append((info["raw_score"], candidate["candidate_id"]))
            n += 1
    return results, n


def pass2_build_rows(path: str, snapshot_date: date, top_ids: set):
    """Re-read the file, recompute full info + reasoning only for top_ids."""
    rows = {}
    with _opener(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # cheap pre-filter without full json parse: candidate_id is the
            # first field, so a substring check is enough to skip ~99.9% of
            # rows before paying the json.loads cost.
            if not any(cid in line for cid in top_ids):
                continue
            candidate = json.loads(line)
            cid = candidate["candidate_id"]
            if cid not in top_ids:
                continue
            info = score_candidate(candidate, snapshot_date)
            reasoning = generate_reasoning(candidate, info)
            rows[cid] = {
                "candidate_id": cid,
                "raw_score": info["raw_score"],
                "reasoning": reasoning,
            }
    return rows


def write_csv(out_path: str, ranked):
    """ranked: list of dicts with candidate_id, raw_score, reasoning,
    already sorted best-first, length == TOP_N."""
    raws = [r["raw_score"] for r in ranked]
    lo, hi = min(raws), max(raws)
    span = (hi - lo) or 1.0

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, row in enumerate(ranked, start=1):
            display_score = 0.40 + 0.59 * (row["raw_score"] - lo) / span
            writer.writerow([
                row["candidate_id"],
                rank,
                f"{display_score:.4f}",
                row["reasoning"],
            ])


def main():
    ap = argparse.ArgumentParser(description="Rank candidates for the Redrob hackathon JD.")
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    ap.add_argument("--out", required=True, help="Path to write the submission CSV")
    args = ap.parse_args()

    t0 = time.time()

    snapshot_date = compute_snapshot_date(args.candidates)
    print(f"[rank.py] snapshot_date (max last_active_date in pool) = {snapshot_date}", file=sys.stderr)

    all_scores, n = pass1_score_all(args.candidates, snapshot_date)
    print(f"[rank.py] scored {n} candidates in {time.time() - t0:.1f}s", file=sys.stderr)

    # Sort by score desc, tie-break candidate_id ascending, take top 100.
    all_scores.sort(key=lambda t: (-t[0], t[1]))
    top = all_scores[:TOP_N]
    top_ids = {cid for _, cid in top}

    rows_by_id = pass2_build_rows(args.candidates, snapshot_date, top_ids)

    ranked = []
    for raw_score, cid in top:
        r = rows_by_id.get(cid)
        if r is None:
            # extremely defensive fallback; should never trigger
            r = {"candidate_id": cid, "raw_score": raw_score, "reasoning": "Top-ranked candidate."}
        ranked.append(r)

    write_csv(args.out, ranked)

    elapsed = time.time() - t0
    print(f"[rank.py] wrote top {len(ranked)} candidates to {args.out} in {elapsed:.1f}s total", file=sys.stderr)


if __name__ == "__main__":
    main()
