from pathlib import Path
from src import config

def test_paths():
    assert isinstance(config.PROJECT_ROOT, Path)
    assert config.SRC_DIR.name == "src"
    assert config.DATA_DIR.name == "requirements"
    assert config.PRECOMPUTED_DIR.name == "precomputed"

def test_model_settings():
    assert config.EMBED_MODEL == "gemini-embedding-2"
    assert config.EMBEDDING_DIM == 768
    assert config.CROSS_ENCODER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"

def test_scoring_weights():
    # Sum of final composite weights should be 1.0 (with float tolerance)
    total_weight = (
        config.WEIGHT_SEMANTIC
        + config.WEIGHT_SKILL
        + config.WEIGHT_CAREER
        + config.WEIGHT_EXPERIENCE
        + config.WEIGHT_BEHAVIOR
        + config.WEIGHT_LOCATION
    )
    assert abs(total_weight - 1.0) < 1e-6
