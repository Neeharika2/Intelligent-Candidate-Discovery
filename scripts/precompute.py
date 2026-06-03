import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.precompute.run_precompute import run_precompute


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run precompute pipeline")
    parser.add_argument("--candidates", type=Path, default=None, help="Path to candidates.jsonl")
    parser.add_argument("--schema", type=Path, default=None, help="Path to candidate_schema.json")
    parser.add_argument("--jd-text", type=Path, default=None, help="Path to JD text file")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for artifacts")
    parser.add_argument("--max-records", type=int, default=None, help="Limit records for quick runs")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_precompute(
        candidates_path=args.candidates,
        schema_path=args.schema,
        jd_text_path=args.jd_text,
        output_dir=args.output_dir,
        max_records=args.max_records,
    )


if __name__ == "__main__":
    main()
