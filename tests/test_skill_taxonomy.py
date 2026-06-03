from src.utils.skill_taxonomy import normalize_skill, get_skill_category, match_core_skills

def test_normalize_skill():
    # Capitalization & spacing
    assert normalize_skill("  Python  ") == "python"
    assert normalize_skill("Sentence-Transformers") == "sentence-transformers"
    assert normalize_skill("vector-db") == "vector database"
    assert normalize_skill("vectordb") == "vector database"
    assert normalize_skill("nlp") == "nlp"
    assert normalize_skill("Natural Language Processing") == "nlp"
    assert normalize_skill("Deep-Learning") == "deep learning"
    assert normalize_skill("MachineLearning") == "machine learning"

def test_get_skill_category():
    assert get_skill_category("sentence-transformers") == "embeddings"
    assert get_skill_category("qdrant") == "retrieval"
    assert get_skill_category("learning-to-rank") == "ranking"
    assert get_skill_category("python") == "programming"
    assert get_skill_category("non-existent-skill") is None

def test_match_core_skills():
    cand_skills = [
        {"name": "Python", "proficiency": "expert", "duration_months": 24, "endorsements": 5},
        {"name": "vector-db", "proficiency": "intermediate", "duration_months": 12, "endorsements": 2},
        {"name": "VectorDB", "proficiency": "advanced", "duration_months": 18, "endorsements": 4}, # Duplicate normalizes to vector database
        {"name": "React", "proficiency": "expert", "duration_months": 36, "endorsements": 10} # Non-core skill
    ]
    core_skills = ["python", "vector database", "faiss"]

    matches = match_core_skills(cand_skills, core_skills)

    # We expect matching "python" and "vector database", and NOT "react" or "faiss"
    assert "python" in matches
    assert "vector database" in matches
    assert "faiss" not in matches
    assert "react" not in matches

    # Check python details
    assert matches["python"]["original_name"] == "Python"
    assert matches["python"]["proficiency"] == "expert"
    assert matches["python"]["duration_months"] == 24

    # Check merged vector database details
    # Max proficiency between intermediate and advanced is advanced
    # Max duration between 12 and 18 is 18
    # Max endorsements between 2 and 4 is 4
    assert matches["vector database"]["proficiency"] == "advanced"
    assert matches["vector database"]["duration_months"] == 18
    assert matches["vector database"]["endorsements"] == 4
