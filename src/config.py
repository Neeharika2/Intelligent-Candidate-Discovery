import os
from pathlib import Path

# ==============================================================================
# 1. Project Directory Structure and Paths
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = PROJECT_ROOT / "sample"
PRECOMPUTED_DIR = PROJECT_ROOT / "precomputed"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"
DOCS_DIR = PROJECT_ROOT / "docs"

# Input / Output files
INPUT_JSONL = DATA_DIR / "candidates.jsonl"
SAMPLE_JSONL = SAMPLE_DIR / "candidates.jsonl"
SAMPLE_JSON = SAMPLE_DIR / "sample_candidates.json"
SCHEMA_PATH = SAMPLE_DIR / "candidate_schema.json"
OUTPUT_DIR = DATA_DIR / "output"
OUTPUT_CSV = OUTPUT_DIR / "submission.csv"

# Precomputed artifact paths
FEATURE_PARQUET_PATH = PRECOMPUTED_DIR / "candidate_features.parquet"
CANDIDATE_EMBEDDINGS_PATH = PRECOMPUTED_DIR / "candidate_embeddings.npy"
SKILL_EMBEDDINGS_PATH = PRECOMPUTED_DIR / "skill_embeddings.npy"
FAISS_INDEX_PATH = PRECOMPUTED_DIR / "faiss_index.bin"
BM25_INDEX_PATH = PRECOMPUTED_DIR / "bm25_index.pkl"
HONEYPOT_FLAGS_PATH = PRECOMPUTED_DIR / "honeypot_flags.json"
CANDIDATE_IDS_PATH = PRECOMPUTED_DIR / "candidate_ids.json"
JD_EMBEDDINGS_PATH = PRECOMPUTED_DIR / "jd_embeddings.json"

# ==============================================================================
# 2. Model Settings
# ==============================================================================
EMBED_MODEL = "gemini-embedding-2"
EMBEDDING_DIM = 768
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBED_BATCH_SIZE = 100

# ==============================================================================
# 3. Retrieval Settings
# ==============================================================================
FAISS_INDEX_TYPE = "IVFFlat"
FAISS_NLIST = 100
FAISS_NPROBE = 10

# Cascade filter top-K parameters
RETRIEVAL_TOP_K = 2000
RERANK_TOP_K = 200
FINAL_TOP_K = 100

# Reciprocal Rank Fusion (RRF) parameters
RRF_K = 60

# ==============================================================================
# 4. Multi-Factor Scoring Weights and Thresholds
# ==============================================================================

# Final composite score weights (must sum to 1.0)
WEIGHT_SEMANTIC = 0.25
WEIGHT_SKILL = 0.25
WEIGHT_CAREER = 0.20
WEIGHT_EXPERIENCE = 0.10
WEIGHT_BEHAVIOR = 0.15
WEIGHT_LOCATION = 0.05

# A. Semantic Fit Component weights (must sum to 1.0)
WEIGHT_SEMANTIC_PROFILE = 0.60
WEIGHT_SEMANTIC_SKILLS = 0.40

# B. Skill Match Component weights (must sum to 1.0)
WEIGHT_SKILL_MUST_HAVE = 0.70
WEIGHT_SKILL_NICE_TO_HAVE = 0.20
WEIGHT_SKILL_ASSESSMENT = 0.10

# Core target skills from the JD
CORE_SKILLS_MUST_HAVE = [
    "embeddings", "sentence-transformers", "retrieval", "vector database",
    "faiss", "qdrant", "pinecone", "milvus", "weaviate", "elasticsearch",
    "python", "ranking", "nlp", "search", "ml", "machine learning"
]

CORE_SKILLS_NICE_TO_HAVE = [
    "lora", "qlora", "peft", "fine-tuning", "xgboost", "learning-to-rank",
    "distributed systems", "llm", "rag", "recommendation systems"
]

# Skill proficiency weights
SKILL_PROFICIENCY_WEIGHTS = {
    "beginner": 0.30,
    "intermediate": 0.60,
    "advanced": 0.85,
    "expert": 1.00
}

# C. Career Trajectory Component weights (must sum to 1.0)
WEIGHT_CAREER_PRODUCT_RATIO = 0.35
WEIGHT_CAREER_DEPLOY_SCORE = 0.30
WEIGHT_CAREER_STABILITY = 0.20
WEIGHT_CAREER_IC_ROLE = 0.15

# Career keywords that signal production experience
DEPLOYMENT_KEYWORDS = [
    "shipped", "deployed", "production", "users", "scale",
    "real-time", "pipeline", "system", "infrastructure", "latency",
    "throughput", "optimization", "operational", "monitoring", "kubernetes"
]

# Roles or keywords that signal non-individual contributor or management track
MANAGEMENT_KEYWORDS = [
    "manager", "vp", "director", "head of", "lead manager", "cto",
    "chief", "product manager", "scrum master", "delivery manager"
]

# D. Experience Fit parameters (Gaussian centered at ideal)
IDEAL_YOE = 7.0
YOE_SIGMA = 3.0

# E. Behavioral / Engagement Component weights (must sum to 1.0)
WEIGHT_BEH_RESPONSE_RATE = 0.30
WEIGHT_BEH_RECENCY = 0.20
WEIGHT_BEH_OPEN_TO_WORK = 0.15
WEIGHT_BEH_RESPONSE_SPEED = 0.10
WEIGHT_BEH_INTERVIEW_RATE = 0.10
WEIGHT_BEH_NOTICE_PERIOD = 0.05
WEIGHT_BEH_PROFILE_COMPLETENESS = 0.05
WEIGHT_BEH_VERIFIED_CONTACT = 0.05  # verified email + phone split equally

# F. Location & Logistics Component weights (must sum to 1.0)
WEIGHT_LOC_COUNTRY = 0.40
WEIGHT_LOC_CITY = 0.30
WEIGHT_LOC_RELOCATE = 0.20
WEIGHT_LOC_WORK_MODE = 0.10

PREFERRED_CITIES = [
    "pune", "noida", "delhi", "gurgaon", "mumbai",
    "hyderabad", "bangalore", "chennai"
]

# ==============================================================================
# 5. Determinism and Sorting
# ==============================================================================
RANDOM_SEED = 42
SORT_KEYS = ["score", "candidate_id"]

# ==============================================================================
# 6. Environment Variables and Keys
# ==============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_ENDPOINT = os.getenv("GEMINI_ENDPOINT", None)
