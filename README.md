# Redrob Hackathon — Candidate Discovery & Ranking

A rule-based, fully-explainable ranker for the Redrob "Intelligent Candidate
Discovery & Ranking Challenge" (Senior AI Engineer — Founding Team JD).

## TL;DR — reproduce the submission

```bash
pip install -r requirements.txt   # optional - core ranker needs no deps at all
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py ./submission.csv   # from the hackathon bundle
```

On the full 100,000-row pool this takes **~25-30 seconds** and **~50 MB of
RAM** on a single CPU core — far inside the 5 min / 16 GB / CPU-only / no
network budget in `submission_spec.docx` Section 3. It also works directly
on a gzipped pool: `--candidates ./candidates.jsonl.gz`.

## Why a rule-based ranker, not embeddings / an LLM re-ranker

The compute constraints (CPU only, no GPU, no network, 5 minutes, for
100,000 rows) explicitly rule out per-candidate LLM calls and make a full
neural re-ranking pass expensive to justify for a one-shot top-100 task.
More importantly: **the JD and the trap design reward correctly reading a
profile, not embedding-similarity to the JD text.** A pure
embedding/keyword-similarity ranker is exactly what the dataset's
"keyword stuffer" and "honeypot" traps are built to defeat (see
`redrob_signals_doc.docx` and the JD's final note to participants). A
small number of **explicit, auditable features**, each traceable to a
specific line in the JD or the signals doc, let us:

1. Defend every weight at the Stage 5 interview.
2. Hit the honeypot-rate constraint deterministically (no honeypot has ever
   appeared in our top 100 across repeated runs on the real pool — see
   "Honeypot detection" below).
3. Stay trivially inside the compute budget.

## How the feature tables were derived

Every constant table in `src/taxonomy.py` is documented in its docstring,
but in short: rather than guessing weights, we ran EDA directly against the
real `candidates.jsonl` (100,000 rows, 47 unique titles, 133 unique skill
names) before writing any scoring code:

- **Title frequency** cleanly separates into 6 tiers by raw count, from
  ~5,500-5,800 occurrences each for 12 off-domain titles (Business Analyst,
  HR Manager, ...) down to just **3-6 occurrences each** for the titles
  that exactly match the JD ("Senior AI Engineer", "Lead AI Engineer",
  "Staff/Senior Machine Learning Engineer", "Senior NLP Engineer", "Senior
  Applied Scientist"). This matches the JD's own framing: *"We're not
  expecting to find many matches in a 100K candidate pool."*
- **Skill frequency** separates into the same kind of tiers: ~12,000-count
  generic skills (HTML, Excel, Sales — present everywhere, no signal),
  ~5,000-count "buzzword" AI skills (LLMs, RAG, Embeddings — present on
  off-domain titles too, i.e. exactly the keyword-stuffer trap), ~1,300-
  count genuine production ML/IR skills (Python, PyTorch, BM25, Learning to
  Rank, Qdrant/Weaviate/Milvus/pgvector, QLoRA/PEFT/LoRA, Elasticsearch/
  OpenSearch), and a long tail of **1-7 occurrence "plain language"**
  skills (Vector Representations, Search Infrastructure, Content Matching,
  Information Retrieval Systems...) that we found correlate almost
  perfectly with the rarest, best-fit candidates — this is the dataset's
  concrete version of the JD's "Tier 5 candidate who doesn't say RAG"
  example.
- **Honeypots**: two cleanly-separated statistical outlier rules were
  found (see `src/honeypot.py` docstring for the exact numbers) and
  account for ~70 of the documented "~80 honeypots", with zero ambiguous
  boundary cases. Crucially, honeypots are deliberately planted *inside*
  high-relevance titles too (we found honeypots tagged "AI Engineer",
  "Senior Machine Learning Engineer", "Search Engineer") — confirming a
  title-only ranker would fail this check.
- **Companies**: the pool mixes 8 generic large fictional employers (no
  signal), the exact consulting firms the JD names plus a few more of the
  same character, and a small set of real Indian/global product & AI-native
  companies (Razorpay, CRED, Zoho, Freshworks, Flipkart, Swiggy, ...).

This EDA is reproducible — see `scripts/eda.py` if included, or just rerun
the snippets described in the module docstrings against your own copy of
`candidates.jsonl`.

## Architecture

```
rank.py                  single CLI entry point (3 streaming passes, see below)
src/
  taxonomy.py             every constant table (title/skill tiers, company
                           lists, location lists, phrase banks) + the EDA
                           that justifies each one
  honeypot.py              explicit honeypot hard-rule
  features.py               8 independent feature extractors, each ~[0,1]
  scoring.py                 weighted combination -> raw_score
  reasoning.py                 builds the 1-2 sentence `reasoning` column
                                 from the SAME fields used for scoring
                                 (zero hallucination by construction)
app.py                    optional Streamlit sandbox (see below)
tests/test_scoring.py     unit tests for honeypot + feature sanity
```

### The 3-pass streaming design (why it's fast and memory-flat)

1. **Pass 0** (regex only, no JSON parsing): scans every `last_active_date`
   value to find the pool's "snapshot date" (max date seen), used as
   "today" for recency scoring. Skipping full JSON parsing here keeps this
   pass extremely cheap.
2. **Pass 1** (full parse, O(1) memory/row): computes the full composite
   score for all 100,000 rows. Only `(score, candidate_id)` is kept per
   row — the parsed dict is discarded immediately. This is the only pass
   that touches the full pool with full feature computation.
3. **Pass 2** (full parse, filtered): re-reads the file and, for just the
   ~100 winning `candidate_id`s, recomputes features and writes the
   `reasoning` text. Cheap pre-filtering avoids paying `json.loads` for
   rows that can't possibly be a match.

This means peak memory never depends on pool size beyond Pass 1's
`(float, str)` list (a few MB even at 1M rows) — measured peak RSS on the
real 100K pool was **~45 MB**.

### Scoring formula

```
base = 0.28*title_score + 0.24*skill_score + 0.12*narrative_score
     + 0.10*experience_score + 0.08*company_score + 0.10*location_score
     + 0.04*education_score

composite = base * behavioral_modifier               # see features.behavioral_modifier
composite *= 0.88   if job_hopper_flag else 1.0
composite *= 0.001  if is_honeypot     else 1.0       # hard kill, see honeypot.py
```

`behavioral_modifier` (range ~0.40-1.25) implements the signals doc's
explicit guidance: *"incorporate them as a multiplier or modifier on top of
skill-match scoring"* — combining `open_to_work_flag`, recency of
`last_active_date`, `recruiter_response_rate`, `profile_completeness_score`,
`notice_period_days`, `interview_completion_rate`, recruiter-demand signals
(`search_appearance_30d`, `saved_by_recruiters_30d`), and `offer_acceptance_rate`.

The final CSV `score` column is a monotonic linear rescale of `raw_score`
into roughly `[0.40, 0.99]` across the top 100 (purely cosmetic — order is
fully determined by `raw_score`, ties broken by `candidate_id` ascending as
required by Section 3 of the spec).

### What this catches, concretely (verified against the real pool)

- 0 of the ~70-80 detected honeypots appear in the produced top 100, even
  though several honeypots carry exactly the titles ("AI Engineer",
  "Senior Machine Learning Engineer") a naive title-matcher would chase.
- Several "Tier 5" exact-title candidates (e.g. "Senior NLP Engineer",
  "Lead AI Engineer") are *correctly excluded* from the top 100 despite a
  perfect title match, because their `recruiter_response_rate` is <0.15
  and/or they've been inactive on-platform for 100-200+ days — i.e. the
  behavioral-signal multiplier is doing real work, exactly as the JD's
  final note asks: *"a perfect-on-paper candidate who hasn't logged in for
  6 months ... is, for hiring purposes, not actually available."*
- Keyword-stuffed off-domain titles (Business Analyst + a pile of LLM/RAG
  buzzwords) score near the bottom of the pool because `title_score`
  dominates the composite and buzzword skills are tenure/trust-gated.

## Limitations / honest caveats

- The weights in `scoring.py` are hand-set from EDA + JD reading, not fit
  against the hidden ground truth (we never had access to it, by design).
  We'd want to validate/tune them against any held-out labels if they ever
  became available.
- `narrative_fit`'s phrase list is a fixed bag-of-phrases, not real NLP — it
  will miss paraphrases it wasn't given and could in principle be gamed by
  a profile that pastes in the right buzzwords inside a job description
  text. We accept this trade-off for the 5-minute/CPU-only/no-network
  budget; a learned text classifier or small local embedding model is the
  natural next step if the compute budget allowed it.
- `job_hopper_flag` is a simple tenure-count heuristic, not a trajectory
  model — it can't distinguish "job-hopping for title bumps" (JD's
  explicit dislike) from "early-career exploration then settling down."

## Sandbox / demo

`app.py` is a small Streamlit app that runs the exact same `src/` pipeline
on an uploaded small candidate sample (≤100 rows), for the Section 10.5
sandbox requirement. Deploy it on Streamlit Community Cloud (free tier):
push this repo to GitHub, then create a new Streamlit Cloud app pointing at
`app.py`. Locally: `streamlit run app.py`.

## Tests

```bash
pip install pytest
pytest tests/ -q
```
(or `python tests/test_scoring.py` if pytest isn't available — it has a
zero-dependency fallback runner at the bottom of the file).

## AI tool usage

Built with substantial use of Claude for code generation, EDA scripting,
and drafting this README. **You (the participant) are responsible for
reading every file in `src/`, understanding why each weight/table exists,
re-running the EDA yourself if you want to double check it, and being able
to defend every design choice in the Stage 5 interview** — per the
hackathon's own framing, AI-assisted work succeeds here only if a human
did real engineering on top of it. Update `submission_metadata.yaml`'s
`ai_usage_summary` honestly before you submit.
