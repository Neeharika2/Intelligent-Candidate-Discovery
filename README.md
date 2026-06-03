# Intelligent Candidate Discovery and Ranking

This repository contains a production-grade plan for ranking the top 100 candidates from a 100K JSONL pool against a single Senior AI Engineer job description.

## What This Project Does

- Builds offline artifacts (features, embeddings, and indexes) using a precompute pipeline.
- Ranks candidates in under 5 minutes on CPU with no network access.
- Produces a CSV with `candidate_id`, `rank`, `score`, and `reasoning`.

## Problem Statement

Rank the top 100 candidates from a 100,000-candidate JSONL pool against a single Senior AI Engineer job description. Output a CSV with `candidate_id`, `rank`, `score`, `reasoning` and score against a hidden ground truth using:

```
Final Composite = 0.50 x NDCG@10 + 0.30 x NDCG@50 + 0.15 x MAP + 0.05 x P@10
```

## Hard Constraints (Ranking Phase)

- Total runtime: <= 5 minutes wall-clock
- Memory: <= 16 GB RAM
- Compute: CPU only (no GPU during ranking)
- Network: off (no external API calls)
- Disk: <= 5 GB intermediate state
- Pre-computation allowed separately (no time limit)

## Dataset Traps

- Honeypot candidates with impossible timelines or skill claims
- Keyword-stuffed profiles with unrelated titles
- Plain-language profiles with real system-building experience
- Honeypot rate > 10% in top 100 is disqualifying

## Why Precompute

The ranking phase has strict constraints: CPU-only, <= 5 minutes, and no network. Precomputing embeddings and indexes allows the ranking stage to run fast and offline.

## Why Central Config

`src/config.py` centralizes paths, weights, and thresholds so runs are reproducible and tuning is safe and consistent.

## Phase Plan

See the detailed phase docs in `docs/`:

- Phase 1: Foundation
- Phase 2: Precompute Pipeline (Network Allowed)
- Phase 3: Ranking Pipeline (Offline Only)
- Phase 4: Reasoning and Output
- Phase 5: Testing and Benchmarks
- Phase 6: Packaging and Submission

## Key Models

- Embeddings: `gemini-embedding-2` (precompute only)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (offline inference)

## Inputs and Outputs

Inputs:
- `candidates.jsonl` (100K candidates)
- `candidate_schema.json` (schema validation)

Outputs:
- `precomputed/` artifacts for offline ranking
- `data/output/submission.csv`

## Expected Constraints

- Ranking runtime: <= 5 minutes wall-clock
- Memory: <= 16 GB
- Network: Off during ranking
- Disk: <= 5 GB intermediate state

## Architecture Overview

The pipeline is a 3-phase cascade with precompute artifacts:

### Phase 0: Pre-Computation (One-Time, Network Allowed)

- Parse and validate all candidates against schema
- Honeypot detection and flagging (do not remove, only flag)
- Feature engineering and text blob generation
- Embedding generation and caching
- Build vector (FAISS) and keyword (BM25) indexes

Artifacts saved:
- `precomputed/candidate_features.parquet`
- `precomputed/candidate_embeddings.npy`
- `precomputed/skill_embeddings.npy`
- `precomputed/jd_embeddings.json`
- `precomputed/faiss_index.bin`
- `precomputed/bm25_index.pkl`
- `precomputed/honeypot_flags.json`
- `precomputed/candidate_ids.json`

### Phase 1: Fast Ranking (<= 5 min, CPU, Offline)

Stage 1: Coarse retrieval (100K -> 2,000)
- Hard filters: remove honeypots, consulting-only careers, no relevant skills
- Vector search over profile embeddings
- BM25 keyword search
- Reciprocal rank fusion to combine results

Stage 2: Detailed scoring (2,000 -> 200)
- Semantic fit
- Skill match
- Career trajectory
- Experience fit
- Behavioral/engagement
- Location/logistics

Stage 3: Precision reranking (200 -> 100)
- Cross-encoder reranking on top 200
- Blend Stage 2 score with cross-encoder score
- Apply final honeypot penalty

Stage 4: Reasoning generation
- Template-based reasoning derived from profile facts only

Stage 5: Output CSV
- `candidate_id`, `rank`, `score`, `reasoning` with validation

## Scoring Summary

Final score is a weighted composite:

```
final_score = (
	0.25 * semantic_score +
	0.25 * skill_score +
	0.20 * career_score +
	0.10 * experience_score +
	0.15 * behavioral_score +
	0.05 * location_score
)
```

Honeypot penalty (safety net):

```
if candidate_id in honeypot_flags:
	final_score *= 0.01
```

## Honeypot Detection (Summary)

- Expert skills with zero duration
- Too many advanced skills for stated years of experience
- Career dates that predate company founding
- Implausible role timelines or overlapping dates

## Notes

This README describes the plan and documentation structure. Implementation steps and production details are in the phase documents under `docs/`.
