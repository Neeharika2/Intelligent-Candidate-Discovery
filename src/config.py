import math

CORE_ML_SKILLS = {
    "machine learning", "deep learning", "neural networks", "pytorch", "tensorflow",
    "scikit-learn", "xgboost", "lightgbm", "model training", "model evaluation",
    "feature engineering", "statistical modeling", "regression", "classification"
}

RETRIEVAL_SKILLS = {
    "information retrieval", "search", "elasticsearch", "opensearch", "solr",
    "bm25", "tf-idf", "ranking", "learning to rank", "recommendation systems",
    "collaborative filtering", "content-based filtering"
}

EMBEDDING_SKILLS = {
    "embeddings", "sentence-transformers", "word2vec", "fasttext", "bert",
    "transformers", "huggingface", "vector search", "semantic search",
    "dense retrieval", "approximate nearest neighbors"
}

VECTOR_DB_SKILLS = {
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "chroma",
    "vector database", "annoy", "scann"
}

LLM_SKILLS = {
    "llm", "large language models", "gpt", "fine-tuning", "fine-tuning llms",
    "lora", "qlora", "peft", "rlhf", "prompt engineering", "rag",
    "retrieval augmented generation", "langchain", "llama", "mistral"
}

NLP_SKILLS = {
    "nlp", "natural language processing", "text classification", "ner",
    "named entity recognition", "sentiment analysis", "text mining",
    "tokenization", "spacy", "nltk"
}

PYTHON_SKILLS = {
    "python", "flask", "fastapi", "django", "pandas", "numpy"
}

WRONG_DOMAIN_SKILLS = {
    "computer vision", "image classification", "object detection", "opencv",
    "speech recognition", "tts", "text to speech", "robotics", "ros",
    "gans", "image segmentation", "yolo", "cnn for images"
}

ALL_RELEVANT_SKILLS = CORE_ML_SKILLS | RETRIEVAL_SKILLS | EMBEDDING_SKILLS | VECTOR_DB_SKILLS | LLM_SKILLS | NLP_SKILLS | PYTHON_SKILLS

WEIGHTS = {
    "technical_fit": 0.45,
    "experience_fit": 0.25,
    "behavioral": 0.20,
    "location": 0.10,
}

POSITIVE_CAREER_PHRASES = [
    "built and deployed", "shipped to production", "production ml",
    "production machine learning", "ranking system", "recommendation system",
    "search infrastructure", "retrieval system", "embedding", "vector search",
    "a/b test", "deployed model", "served model", "real-time inference",
    "training pipeline", "feature store", "model monitoring",
    "end-to-end ml", "ml pipeline", "data pipeline for ml",
    "fine-tuned", "fine-tuning", "model evaluation", "ndcg", "mrr",
    "real users", "production deployment", "latency optimization",
    "scaled to", "millions of", "inference", "batch prediction",
]

NEGATIVE_CAREER_PHRASES = [
    "my own technical depth in ai is limited",
    "technical depth is limited", "limited technical",
    "experimenting with chatgpt", "experimented with",
    "ai-strategy advisory", "ai-assisted content",
    "curious about how ai", "exploring ai",
    "adjacent to ml", "lighter on technical depth",
    "no production deployment", "self-directed ml projects",
    "completed a couple of", "building competence",
]

CONSULTING_INDUSTRIES = {
    "IT Services", "Consulting", "Professional Services", "Staffing", "Outsourcing"
}

PREFERRED_CITIES = {"pune", "noida"}
TIER1_INDIAN_CITIES = {
    "hyderabad", "mumbai", "delhi", "delhi ncr", "bangalore",
    "bengaluru", "gurgaon", "gurugram", "chennai", "kolkata"
}

CITY_ALIASES = {
    "gurgaon": "gurugram",
    "bangalore": "bengaluru",
    "new delhi": "delhi",
    "national capital region": "delhi ncr",
    "ncr": "delhi ncr",
}

RETRIEVAL_TOP_K_BM25 = 2000
RETRIEVAL_TOP_K_FAISS = 2000
RRF_K = 60

RERANK_TOP_N = 500
RERANK_BATCH_SAVE_EVERY = 50
RERANK_MODEL = "deepseek-v4-flash"

DEEPSEEK_API_KEY = None  # Set from env or .env
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

EMBEDDING_MODEL = "bge-m3"
EMBEDDING_BATCH_SIZE = 4
EMBEDDING_CHECKPOINT_EVERY = 10

BGE_M3_API_URL = "http://192.168.31.246:5001/embed"
BGE_M3_EMBEDDING_DIM = 1024
BGE_M3_REQUEST_TIMEOUT = 60
BGE_M3_MAX_RETRIES = 5

SCORE_TIERS = [
    (0.75, 1.00, 0.85, 1.00),
    (0.50, 0.75, 0.55, 0.85),
    (0.25, 0.50, 0.25, 0.55),
    (0.00, 0.25, 0.00, 0.25),
]


SKILL_ALIASES = {
    "rag": "retrieval augmented generation",
    "llms": "large language models",
    "large language model": "large language models",
    "nlp": "natural language processing",
    "ml": "machine learning",
    "dl": "deep learning",
    "cv": "computer vision",
    "k8s": "kubernetes",
    "tf": "tensorflow",
    "sklearn": "scikit-learn",
    "sk learn": "scikit-learn",
    "hf": "huggingface",
    "hugging face": "huggingface",
    "ann": "approximate nearest neighbors",
    "ltr": "learning to rank",
    "ir": "information retrieval",
}


def normalize_skill_name(name: str) -> str:
    n = name.lower().strip()
    n = n.replace("-", " ").replace("_", " ")
    return SKILL_ALIASES.get(n, n)


def skill_matches_taxonomy(skill_name: str, taxonomy: set) -> bool:
    normalized = normalize_skill_name(skill_name)
    for term in taxonomy:
        if normalized == term or term in normalized or normalized in term:
            return True
    return False


def count_taxonomy_skills(skills: list, taxonomy: set) -> float:
    total = 0.0
    for s in skills:
        if skill_matches_taxonomy(s["name"], taxonomy):
            prof_map = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0}
            duration_factor = min(s.get("duration_months", 0) / 36.0, 1.0)
            total += prof_map.get(s.get("proficiency", "beginner"), 0.25) * duration_factor
    return total


def normalize_city(city: str) -> str:
    if not city:
        return ""
    c = city.lower().split(",")[0].strip()
    for alias, canonical in CITY_ALIASES.items():
        if c == alias:
            return canonical
    return c


def gaussian_score(value: float, center: float, sigma: float, max_score: float) -> float:
    return max_score * math.exp(-0.5 * ((value - center) / sigma) ** 2)