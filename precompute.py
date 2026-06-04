import argparse
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.pipeline import run_precompute


def main():
    parser = argparse.ArgumentParser(description="Pre-compute artifacts for candidate ranking")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--jd", required=True, help="Path to job description text file")
    parser.add_argument("--out", default="./artifacts", help="Output directory for artifacts")
    parser.add_argument("--deepseek-api-key", default=None, help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)")
    args = parser.parse_args()

    deepseek_api_key = args.deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY")

    if not deepseek_api_key:
        print("Warning: No DeepSeek API key provided. Set --deepseek-api-key or DEEPSEEK_API_KEY env var.")
        print("Ideal profile extraction and re-ranking will fail without a DeepSeek API key.")
        sys.exit(1)

    run_precompute(
        candidates_path=args.candidates,
        jd_path=args.jd,
        output_dir=args.out,
        deepseek_api_key=deepseek_api_key,
    )


if __name__ == "__main__":
    main()