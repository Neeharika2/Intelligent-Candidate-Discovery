import json
from pathlib import Path
from src.utils.logging import StageLogger, calculate_checksum, create_artifact_metadata

def test_stage_logger(tmp_path):
    log_file = tmp_path / "stage_run.jsonl"
    logger = StageLogger(log_path=log_file)
    
    logger.log_stage("retrieve", 1.25, 2000, {"index_type": "faiss"})
    logger.log_stage("score", 0.75, 200, {"mode": "hybrid"})
    
    assert log_file.exists()
    
    # Read and parse lines
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 2
    
    entry1 = json.loads(lines[0])
    assert entry1["stage"] == "retrieve"
    assert entry1["duration_sec"] == 1.25
    assert entry1["candidates_count"] == 2000
    assert entry1["metadata"]["index_type"] == "faiss"
    assert "timestamp" in entry1
    
    entry2 = json.loads(lines[1])
    assert entry2["stage"] == "score"
    assert entry2["duration_sec"] == 0.75
    assert entry2["candidates_count"] == 200
    assert entry2["metadata"]["mode"] == "hybrid"

def test_checksum_and_metadata(tmp_path):
    dummy_file = tmp_path / "artifact.npy"
    dummy_file.write_text("dummy embedding content")
    
    checksum = calculate_checksum(dummy_file)
    assert len(checksum) == 64 # SHA-256 is 64 hex characters
    
    metadata = create_artifact_metadata("gemini-embedding-2", dummy_file)
    assert metadata["model_name"] == "gemini-embedding-2"
    assert metadata["checksum"] == checksum
    assert "timestamp" in metadata
