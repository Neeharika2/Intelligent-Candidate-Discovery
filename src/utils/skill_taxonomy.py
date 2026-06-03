import re
from typing import List, Dict, Any, Optional

# ==============================================================================
# Skill Synonym Mapping
# ==============================================================================
# Keys in this map should be fully cleaned (lowercase, alphanumeric, single spaces)
SYNONYM_MAP = {
    "vector db": "vector database",
    "vectordb": "vector database",
    "vector databases": "vector database",
    "vector search": "vector database",
    "faiss": "faiss",
    "facebook AI similarity search": "faiss",
    "qdrant": "qdrant",
    "pinecone": "pinecone",
    "milvus": "milvus",
    "weaviate": "weaviate",
    "elasticsearch": "elasticsearch",
    "elastic search": "elasticsearch",
    "sentence transformers": "sentence-transformers",
    "sentencetransformers": "sentence-transformers",
    "sentence transformer": "sentence-transformers",
    "nlp": "nlp",
    "natural language processing": "nlp",
    "ml": "machine learning",
    "machinelearning": "machine learning",
    "deeplearning": "deep learning",
    "deep learning": "deep learning",
    "llm": "llm",
    "llms": "llm",
    "large language model": "llm",
    "large language models": "llm",
    "rag": "rag",
    "retrieval augmented generation": "rag",
    "retrieval-augmented generation": "rag",
    "fine tuning": "fine-tuning",
    "finetuning": "fine-tuning",
    "lora": "lora",
    "qlora": "qlora",
    "peft": "peft",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "lgb": "lightgbm",
    "learning to rank": "learning-to-rank",
    "learningtorank": "learning-to-rank",
    "ltr": "learning-to-rank",
    "distributed systems": "distributed systems",
    "recommendation systems": "recommendation systems",
    "recsys": "recommendation systems",
    "python": "python",
    "python3": "python",
    "ranking": "ranking",
    "search": "search",
    "information retrieval": "retrieval",
    "ir": "retrieval"
}

# ==============================================================================
# Skill Group / Bucket Mapping
# ==============================================================================
SKILL_GROUPS = {
    "embeddings": [
        "embeddings", "sentence-transformers", "bert", "word2vec", 
        "cohere", "openai embeddings", "text embeddings"
    ],
    "retrieval": [
        "retrieval", "vector database", "faiss", "qdrant", "pinecone", 
        "milvus", "weaviate", "elasticsearch", "hybrid search", "bm25"
    ],
    "ranking": [
        "ranking", "learning-to-rank", "cross-encoder", "reranking", 
        "ndcg", "mrr", "map", "xgboost", "lightgbm"
    ],
    "nlp": [
        "nlp", "llm", "rag", "fine-tuning", "lora", "qlora", "peft", 
        "transformers", "spacy", "nltk", "langchain", "llama-index"
    ],
    "programming": [
        "python", "pytorch", "tensorflow", "keras", "scikit-learn", 
        "numpy", "pandas", "sql", "spark", "airflow", "fastapi", 
        "flask", "docker", "git"
    ]
}

def normalize_skill(skill_name: str) -> str:
    """
    Cleans and normalizes a skill name by:
    1. Lowercasing and trimming whitespace.
    2. Replacing punctuation (hyphens, underscores, slashes, periods) with spaces.
    3. Collapsing multiple spaces.
    4. Stripping all non-alphanumeric characters (except spaces).
    5. Mapping to a canonical name if found in the synonym map.
    """
    if not skill_name:
        return ""
    
    # Lowercase
    cleaned = skill_name.lower().strip()
    
    # Replace standard delimiters with space
    cleaned = re.sub(r'[-_/\.,]', ' ', cleaned)
    
    # Remove non-alphanumeric (keep spaces)
    cleaned = re.sub(r'[^a-z0-9\s]', '', cleaned)
    
    # Collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Resolve synonym
    return SYNONYM_MAP.get(cleaned, cleaned)

def get_skill_category(canonical_name: str) -> Optional[str]:
    """
    Returns the group/category for a given canonical skill name, or None if not found.
    """
    for category, skills in SKILL_GROUPS.items():
        if canonical_name in skills:
            return category
    return None

def match_core_skills(candidate_skills: List[Dict[str, Any]], core_skills_list: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Processes a list of candidate skills and matches them against a list of target core skills.
    Handles duplicate candidate skills that normalize to the same canonical representation
    by selecting the one with the maximum proficiency and maximum duration.

    Args:
        candidate_skills: List of dicts, e.g. [{"name": "Python", "proficiency": "expert", "duration_months": 24}]
        core_skills_list: List of target skill strings, e.g. ["python", "vector database"]

    Returns:
        Dict mapping canonical matched skill names to their details:
        {
            "canonical_name": {
                "original_name": str,
                "proficiency": str,
                "duration_months": int,
                "endorsements": int
            }
        }
    """
    normalized_targets = {normalize_skill(skill) for skill in core_skills_list}
    matched = {}

    # Define ordering for proficiency comparison
    prof_order = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}

    for skill in candidate_skills:
        raw_name = skill.get("name", "")
        norm_name = normalize_skill(raw_name)

        if norm_name in normalized_targets:
            proficiency = skill.get("proficiency", "beginner").lower()
            if proficiency not in prof_order:
                proficiency = "beginner"
                
            duration = skill.get("duration_months", 0)
            endorsements = skill.get("endorsements", 0)

            if norm_name not in matched:
                matched[norm_name] = {
                    "original_name": raw_name,
                    "proficiency": proficiency,
                    "duration_months": duration,
                    "endorsements": endorsements
                }
            else:
                # Merge logic: take max proficiency and max duration
                existing = matched[norm_name]
                if prof_order[proficiency] > prof_order[existing["proficiency"]]:
                    existing["proficiency"] = proficiency
                    existing["original_name"] = raw_name
                
                existing["duration_months"] = max(existing["duration_months"], duration)
                existing["endorsements"] = max(existing["endorsements"], endorsements)

    return matched
