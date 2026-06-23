"""
taxonomy.py
-----------
All "domain knowledge" constant tables used by the ranker, derived from:
  (a) a close reading of job_description.docx / redrob_signals_doc.docx /
      submission_spec.docx, and
  (b) empirical EDA over the actual 100,000-row candidates.jsonl pool
      (title frequency table, skill frequency table, company frequency
      table, and outlier analysis for honeypot detection).

Putting these in one file makes the scoring logic in scoring.py declarative
and makes it easy to defend/explain every weight at the Stage 5 interview.

EDA notes (see README.md "How these tables were derived" for the full
methodology):

  TITLE FREQUENCY (100K pool, 47 unique titles total):
    Tier 0 (off-domain, ~5500-5800 each):  Business Analyst, HR Manager, ...
    Tier 1 (general SWE, ~2700-3450 each): Software Engineer, QA Engineer, ...
    Tier 2 (data/senior-adjacent, ~650-770 each): Data Engineer, Backend Eng, ...
    Tier 3 (core ML/AI, ~130-170 each): ML Engineer, Data Scientist, ...
    Tier 4 (senior/specialised ML-IR, ~14-26 each): AI Engineer, Search Eng, ...
    Tier 5 (exact-role match, only 3-6 each!): Senior AI Engineer, Lead AI
        Engineer, Staff/Senior Machine Learning Engineer, Senior NLP
        Engineer, Senior Applied Scientist.
    This matches the JD's own framing: "We're not expecting to find many
    matches in a 100K candidate pool. We're explicitly OK with that."

  SKILL FREQUENCY (133 unique skill names across the pool):
    ~11800-12250 count: generic, cross-domain skills (HTML, Excel, Sales,
        Java, AWS, ...). Present across almost every title bucket including
        the off-domain ones -> close to zero discriminative signal here.
    ~4600-5200 count: "buzzword" AI/ML skills (LLMs, RAG, Embeddings,
        LangChain, Computer Vision, MLOps, ...). High enough frequency that
        they show up attached to off-domain titles too (the JD's explicit
        "keyword stuffer" trap) -> must be trust-gated by tenure, not just
        counted.
    ~1280-1400 count: genuine production ML/IR/infra skills (Python,
        PyTorch, scikit-learn, BM25, Learning to Rank, Elasticsearch,
        Qdrant/Weaviate/Milvus/pgvector, QLoRA/PEFT/LoRA, TensorFlow,
        Haystack, LlamaIndex, OpenSearch). Frequency tracks almost exactly
        with the count of genuinely ML-titled candidates -> strong signal.
    1-7 count: rare "plain language" paraphrases (Vector Representations,
        Content Matching, Information Retrieval Systems, Workflow
        Orchestration, Search Infrastructure, Search & Discovery, Ranking
        Systems, Indexing Algorithms, Text Encoders, Search Backend,
        Natural Language Processing, Document Processing, Model Adaptation,
        Open-source ML libraries). These almost perfectly identify the
        hand-written "Tier 5" golden candidates described in the JD's final
        note ("a Tier 5 candidate may not use the words RAG or Pinecone").

  HONEYPOTS: two cleanly-separated (no false positives at the boundary)
    statistical outlier patterns were found and are used as a hard
    disqualifying rule in honeypot.py:
      1. >=1 skill marked "expert" proficiency with 0 months duration used
         (21 candidates, vs. 0 candidates with 1-2 such skills - a sharp,
         deliberately-injected discontinuity).
      2. sum(career_history.duration_months)/12 vs profile.years_of_experience
         ratio outside [0.5, 1.5] (49 candidates; population is otherwise
         tightly clustered in [0.94, 1.0], so the cut is not arbitrary).
    These two groups are disjoint (0 overlap) and together account for ~70
    of the documented "~80 honeypots" -> good confidence in this rule.
    Crucially, honeypots are NOT confined to off-domain titles - several
    carry titles like "AI Engineer" / "Senior Machine Learning Engineer" /
    "Search Engineer", i.e. they are specifically designed to defeat a
    title-only ranker.

  COMPANIES: the pool contains a deliberate mix of (a) 8 generic large
    fictional employers (Acme Corp, Globex Inc, Initech, Hooli, ...) used
    as filler with no special signal, (b) the exact consulting firms the
    JD names (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) plus a
    few more of the same character (HCL, Tech Mahindra, Mindtree, Mphasis),
    and (c) a small set of real Indian/global product & AI-native companies
    (Razorpay, CRED, Zoho, Freshworks, Flipkart, Swiggy, Zomato, Meesho,
    InMobi, Nykaa, Ola, Vedantu) that line up with the JD's "product
    company, not pure services" preference.
"""

# ---------------------------------------------------------------------------
# 1. TITLE TIERS
# ---------------------------------------------------------------------------
# Score contribution per tier (0..1). Tier 5 = exact role match (rarest).
TITLE_TIER_SCORE = {
    5: 1.00,
    4: 0.84,
    3: 0.66,
    2: 0.44,
    1: 0.22,
    0: 0.06,
}

TITLE_TIER = {
    # Tier 5 - exact / near-exact match to "Senior AI Engineer" (rarest titles in pool)
    "Senior AI Engineer": 5,
    "Lead AI Engineer": 5,
    "Senior Applied Scientist": 5,
    "Staff Machine Learning Engineer": 5,
    "Senior Machine Learning Engineer": 5,
    "Senior NLP Engineer": 5,
    # Tier 4 - senior / specialised ML-IR roles
    "Senior Data Scientist": 4,
    "NLP Engineer": 4,
    "AI Engineer": 4,
    "Applied ML Engineer": 4,
    "Search Engineer": 4,
    "Machine Learning Engineer": 4,
    "Recommendation Systems Engineer": 4,
    # Tier 3 - core ML/AI, not necessarily senior
    "AI Specialist": 3,
    "Junior ML Engineer": 3,
    "Computer Vision Engineer": 3,
    "Senior Software Engineer (ML)": 3,
    "Data Scientist": 3,
    "AI Research Engineer": 3,
    "ML Engineer": 3,
    # Tier 2 - data/backend, senior-adjacent
    "Senior Software Engineer": 2,
    "Senior Data Engineer": 2,
    "Backend Engineer": 2,
    "Data Analyst": 2,
    "Data Engineer": 2,
    "Analytics Engineer": 2,
    # Tier 1 - general software engineering (not ML/IR specific)
    "QA Engineer": 1,
    "Frontend Engineer": 1,
    "Mobile Developer": 1,
    "DevOps Engineer": 1,
    ".NET Developer": 1,
    "Java Developer": 1,
    "Cloud Engineer": 1,
    "Full Stack Developer": 1,
    "Software Engineer": 1,
    # Tier 0 - off-domain (the bulk-filler "irrelevant" titles)
    "Business Analyst": 0,
    "HR Manager": 0,
    "Mechanical Engineer": 0,
    "Accountant": 0,
    "Project Manager": 0,
    "Customer Support": 0,
    "Operations Manager": 0,
    "Content Writer": 0,
    "Sales Executive": 0,
    "Civil Engineer": 0,
    "Graphic Designer": 0,
    "Marketing Manager": 0,
}
DEFAULT_TITLE_TIER = 1  # unseen/unknown title -> treat as generic SWE, neutral-low

# ---------------------------------------------------------------------------
# 2. SKILL TIERS
# ---------------------------------------------------------------------------
# tier weight multipliers applied on top of proficiency * tenure-trust
CORE_SKILL_WEIGHT = 1.00
BUZZWORD_SKILL_WEIGHT = 0.55
RARE_SKILL_WEIGHT = 1.65

CORE_SKILLS = {
    "Python", "PyTorch", "TensorFlow", "scikit-learn", "NLP",
    "Machine Learning", "Deep Learning", "Elasticsearch", "OpenSearch",
    "BM25", "Learning to Rank", "Qdrant", "Weaviate", "Milvus", "pgvector",
    "QLoRA", "PEFT", "LoRA", "Haystack", "LlamaIndex",
}

BUZZWORD_SKILLS = {
    "Hugging Face Transformers", "LangChain", "Information Retrieval",
    "LLMs", "Recommendation Systems", "Semantic Search",
    "Sentence Transformers", "Embeddings", "Vector Search",
    "Prompt Engineering", "Pinecone", "FAISS", "RAG", "Fine-tuning LLMs",
    "YOLO", "GANs", "Feature Engineering", "OpenCV", "ASR",
    "Image Classification", "Computer Vision", "Speech Recognition", "CNN",
    "Kubeflow", "MLOps", "BentoML", "Data Science", "Reinforcement Learning",
    "Object Detection", "Diffusion Models", "MLflow", "Time Series",
    "Weights & Biases", "Forecasting", "TTS", "Statistical Modeling",
}

RARE_SKILLS = {
    "Information Retrieval Systems", "Search Backend", "Text Encoders",
    "Vector Representations", "Content Matching", "Model Adaptation",
    "Ranking Systems", "Search & Discovery", "Workflow Orchestration",
    "Search Infrastructure", "Indexing Algorithms",
    "Open-source ML libraries", "Natural Language Processing",
    "Document Processing",
}

PROFICIENCY_MULT = {
    "beginner": 0.30,
    "intermediate": 0.55,
    "advanced": 0.80,
    "expert": 1.00,
}

# "Must have" skills the JD calls out by name - used for a small explicit bonus
JD_NAMED_SKILLS = {
    "Embeddings", "Sentence Transformers", "Pinecone", "Weaviate", "Qdrant",
    "Milvus", "OpenSearch", "Elasticsearch", "FAISS", "Python",
    "Learning to Rank", "RAG",
}

# ---------------------------------------------------------------------------
# 3. COMPANIES / INDUSTRIES
# ---------------------------------------------------------------------------
CONSULTING_FIRMS = {
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini",
    "HCL", "Tech Mahindra", "Mindtree", "Mphasis",
}

AI_NATIVE_INDUSTRIES = {"AI/ML", "AI Services", "HealthTech AI", "Conversational AI"}

PRODUCT_COMPANIES = {
    "Razorpay", "CRED", "Zoho", "Freshworks", "Flipkart", "Swiggy",
    "Zomato", "Meesho", "InMobi", "Nykaa", "Ola", "Vedantu",
}

# ---------------------------------------------------------------------------
# 4. LOCATION
# ---------------------------------------------------------------------------
# Cities the JD explicitly names as the office locations / welcome list.
PRIMARY_CITIES = {"pune", "noida"}
WELCOME_CITIES = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore", "bengaluru",
}

# ---------------------------------------------------------------------------
# 5. CAREER-NARRATIVE ("PRODUCTION SHIPPING") PHRASES
# ---------------------------------------------------------------------------
# Mined from career_history.description / profile.summary text. Presence of
# these phrases is a much stronger "this person actually did the work"
# signal than a skills-list keyword, and is exactly how the JD says it wants
# to be read: "the gap between what the JD says and what the JD means."
PRODUCTION_PHRASES = [
    "shipped", "production", "real users", "at scale", "deployed",
    "a/b test", "ab test", "offline benchmark", "evaluation framework",
    "ndcg", "mrr", "map@", "recall@", "precision@", "latency", "throughput",
    "drift", "retrain", "ranking layer", "retrieval system",
    "search infrastructure", "matching layer", "personalization",
    "query understanding", "index refresh", "feature pipeline",
    "online experiment", "recommendation system", "search and discovery",
    "millions of", "billions of", "real-time", "low latency",
    "hybrid retrieval", "vector search", "embedding drift",
]

# Phrases that explicitly signal the JD's stated disqualifiers
PURE_RESEARCH_PHRASES = [
    "research-only", "research only", "purely theoretical",
    "no production deployment", "never shipped", "never deployed",
    "academic lab", "theoretical research",
]
