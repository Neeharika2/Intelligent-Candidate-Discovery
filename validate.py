import argparse
import json
import sys

from src.loader import load_candidates
from src.honeypot import detect_honeypot


def validate_format(submission_path: str) -> list[str]:
    import pandas as pd
    errors = []
    try:
        df = pd.read_csv(submission_path)
    except Exception as e:
        return [f"Cannot read CSV: {e}"]

    if len(df) != 100:
        errors.append(f"Expected 100 rows, got {len(df)}")

    required_cols = ["candidate_id", "rank", "score", "reasoning"]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if "rank" in df.columns:
        ranks = sorted(df["rank"].tolist())
        if ranks != list(range(1, 101)):
            errors.append(f"Ranks should be 1-100, got range {min(ranks)}-{max(ranks)}")

    if "score" in df.columns:
        scores = df["score"].tolist()
        for i in range(len(scores) - 1):
            if scores[i] < scores[i + 1]:
                errors.append(f"Scores not descending: rank {i+1} score {scores[i]} < rank {i+2} score {scores[i+1]}")
                break

    if "candidate_id" in df.columns:
        if len(df["candidate_id"].unique()) != 100:
            errors.append(f"Duplicate candidate_ids found")

    return errors


def validate_archetypes(submission_path: str, candidates: list[dict]) -> list[str]:
    import pandas as pd
    warnings = []
    df = pd.read_csv(submission_path)
    cands_by_id = {c["candidate_id"]: c for c in candidates}

    for _, row in df.head(50).iterrows():
        cid = row["candidate_id"]
        if cid not in cands_by_id:
            continue
        c = cands_by_id[cid]
        title = (c.get("profile", {}).get("current_title", "") or "").lower()
        industry = c.get("profile", {}).get("current_industry", "")

        non_tech = ["hr manager", "accountant", "marketing manager", "graphic designer",
                     "content writer", "operations manager", "sales executive", "customer support"]
        if any(nt in title for nt in non_tech):
            if industry in ["IT Services", "Consulting"]:
                warnings.append(f"Row {row['rank']}: {cid} is {title} at consulting firm — likely should not be top 50")

    ml_in_top_20 = 0
    for _, row in df.head(20).iterrows():
        cid = row["candidate_id"]
        if cid not in cands_by_id:
            continue
        c = cands_by_id[cid]
        title = (c.get("profile", {}).get("current_title", "") or "").lower()
        industry = c.get("profile", {}).get("current_industry", "")
        ml_keywords = ["ml engineer", "ai engineer", "data scientist", "machine learning"]
        if any(kw in title for kw in ml_keywords) and industry not in ["IT Services", "Consulting"]:
            ml_in_top_20 += 1

    if ml_in_top_20 == 0:
        warnings.append("No ML/AI engineers from product companies in top 20 — check scoring logic")

    return warnings


def validate_honeypots(submission_path: str, candidates: list[dict]) -> list[str]:
    import pandas as pd
    warnings = []
    df = pd.read_csv(submission_path)
    cands_by_id = {c["candidate_id"]: c for c in candidates}

    honeypot_count = 0
    for _, row in df.iterrows():
        cid = row["candidate_id"]
        if cid not in cands_by_id:
            continue
        c = cands_by_id[cid]
        flagged, reason = detect_honeypot(c)
        if flagged:
            honeypot_count += 1

    if honeypot_count > 10:
        warnings.append(f"CRITICAL: {honeypot_count} honeypots in top 100 — risk of disqualification (limit is 10%)")
    elif honeypot_count > 0:
        warnings.append(f"Warning: {honeypot_count} honeypots in top 100")

    return warnings


def main():
    parser = argparse.ArgumentParser(description="Validate submission CSV")
    parser.add_argument("--submission", required=True, help="Path to submission CSV")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    args = parser.parse_args()

    print("=== Format Validation ===")
    errors = validate_format(args.submission)
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
    else:
        print("  All format checks passed!")

    print("\n=== Archetype Validation ===")
    candidates = load_candidates(args.candidates)
    warnings = validate_archetypes(args.submission, candidates)
    if warnings:
        for w in warnings:
            print(f"  WARNING: {w}")
    else:
        print("  No archetype warnings!")

    print("\n=== Honeypot Check ===")
    hp_warnings = validate_honeypots(args.submission, candidates)
    for w in hp_warnings:
        print(f"  {w}")

    has_errors = len(errors) > 0
    sys.exit(1 if has_errors else 0)


if __name__ == "__main__":
    main()