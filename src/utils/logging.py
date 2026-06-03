import json
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

class StageLogger:
    """
    A lightweight logger that outputs stage execution times, candidate counts,
    and metadata in JSON Lines (JSONL) format for structured pipeline observability.
    """
    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            # Ensure the directory exists
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            
    def log_stage(self, stage: str, duration_sec: float, candidates_count: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Logs a stage's execution performance.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "duration_sec": round(duration_sec, 4),
            "candidates_count": candidates_count,
            "metadata": metadata or {}
        }
        
        # Format as JSON line
        log_line = json.dumps(entry)
        
        # Always print to stdout for real-time tracking
        print(log_line, flush=True)
        
        # Append to log file if configured
        if self.log_path:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(log_line + "\n")
            except Exception as e:
                print(f"Error writing to log file {self.log_path}: {e}", flush=True)

def calculate_checksum(file_path: Path) -> str:
    """
    Computes the SHA-256 checksum of a file for artifact verification.
    """
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256.update(byte_block)
        return sha256.hexdigest()
    except FileNotFoundError:
        return ""

def create_artifact_metadata(model_name: str, file_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Generates standard artifact metadata including timestamp, model name, and file checksum.
    """
    checksum = ""
    if file_path and file_path.exists():
        checksum = calculate_checksum(file_path)
        
    return {
        "model_name": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checksum": checksum
    }
