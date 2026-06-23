"""
app.py
------
Minimal Streamlit sandbox so organizers (and we) can verify the ranker runs
end-to-end on a small sample, per submission_spec.docx Section 10.5.

Deploy: push this repo to GitHub, then deploy on Streamlit Community Cloud
pointing at this file. Free tier is sufficient - the ranker itself needs no
GPU, no network, and trivial RAM (it streams JSON Lines and never holds
more than ~100 parsed candidates in memory at once).

Usage:
    streamlit run app.py
Then upload a small candidates.jsonl sample (<=100 candidates is plenty -
sample_candidates.json from the hackathon bundle works after conversion,
or just a handful of lines copied out of the full candidates.jsonl).
"""

import io
import json
from datetime import date

import streamlit as st

from src.scoring import score_candidate
from src.reasoning import generate_reasoning

st.set_page_config(page_title="Redrob Hackathon - Candidate Ranker Sandbox", layout="wide")

st.title("Redrob Hackathon — Candidate Ranker Sandbox")
st.markdown(
    "Upload a small sample of candidates (JSON Lines, one candidate object per "
    "line — same schema as `candidates.jsonl`) and this will run the exact "
    "same scoring pipeline used to produce the full submission, entirely "
    "on CPU with no network calls."
)

uploaded = st.file_uploader(
    "Candidate sample (.jsonl, ideally <= 100 rows for a fast demo)",
    type=["jsonl", "json", "txt"],
)

if uploaded is not None:
    raw = uploaded.read().decode("utf-8")
    candidates = []
    stripped = raw.strip()
    try:
        if stripped.startswith("["):
            candidates = json.loads(stripped)
        else:
            for line in stripped.splitlines():
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    except json.JSONDecodeError as e:
        st.error(f"Could not parse file as JSON / JSON Lines: {e}")
        st.stop()

    st.success(f"Loaded {len(candidates)} candidate(s).")

    # Snapshot date = max last_active_date seen in this sample (same logic
    # as rank.py's Pass 0, just done in-memory since the sample is small).
    dates = []
    for c in candidates:
        d = c.get("redrob_signals", {}).get("last_active_date")
        if d:
            try:
                dates.append(date.fromisoformat(d))
            except ValueError:
                pass
    snapshot_date = max(dates) if dates else date.today()
    st.caption(f"Using snapshot date: {snapshot_date}")

    rows = []
    for c in candidates:
        info = score_candidate(c, snapshot_date)
        reasoning = generate_reasoning(c, info)
        rows.append({
            "candidate_id": c.get("candidate_id"),
            "raw_score": info["raw_score"],
            "is_honeypot": info["is_honeypot"],
            "current_title": info.get("current_title"),
            "years_of_experience": info.get("years_of_experience"),
            "reasoning": reasoning,
        })

    rows.sort(key=lambda r: (-r["raw_score"], r["candidate_id"]))

    if rows:
        lo = rows[-1]["raw_score"]
        hi = rows[0]["raw_score"]
        span = (hi - lo) or 1.0
        for i, r in enumerate(rows, start=1):
            r["rank"] = i
            r["score"] = round(0.40 + 0.59 * (r["raw_score"] - lo) / span, 4)

    st.subheader("Ranked output")
    st.dataframe(
        [{"rank": r["rank"], "candidate_id": r["candidate_id"], "score": r["score"],
          "title": r["current_title"], "honeypot": r["is_honeypot"],
          "reasoning": r["reasoning"]} for r in rows],
        use_container_width=True,
    )

    csv_buf = io.StringIO()
    csv_buf.write("candidate_id,rank,score,reasoning\n")
    for r in rows:
        reasoning_escaped = '"' + r["reasoning"].replace('"', '""') + '"'
        csv_buf.write(f"{r['candidate_id']},{r['rank']},{r['score']:.4f},{reasoning_escaped}\n")

    st.download_button(
        "Download ranked CSV",
        data=csv_buf.getvalue(),
        file_name="sandbox_ranked_sample.csv",
        mime="text/csv",
    )
else:
    st.info("Waiting for a file upload. Try the bundle's `sample_candidates.json` "
            "(it's a JSON array - this app handles both array and JSONL input).")
