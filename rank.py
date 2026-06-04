import argparse
import os
import sys

from src.pipeline import run_ranking


def main():
    parser = argparse.ArgumentParser(description="Rank candidates for the Redrob hackathon")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--artifacts", default="./artifacts", help="Path to pre-computed artifacts directory")
    parser.add_argument("--out", default="./submission.csv", help="Output CSV path")
    args = parser.parse_args()

    if not os.path.exists(args.artifacts):
        print(f"Error: Artifacts directory not found: {args.artifacts}")
        print("Run precompute.py first to generate artifacts.")
        sys.exit(1)

    run_ranking(
        artifacts_dir=args.artifacts,
        candidates_path=args.candidates,
        output_csv=args.out,
    )


if __name__ == "__main__":
    main()