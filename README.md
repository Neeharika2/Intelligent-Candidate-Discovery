# Intelligent Candidate Discovery & Ranking Solution
This repository contains the candidate ranking and discovery system built for the Redrob Hackathon. The system ranks 100,000 candidates against a Senior AI Engineer job description and filters out honeypots (impossible profiles), outputs the top 100 with LLM-grade reasoning, and runs within compute constraints.

## System Architecture & Features
* **Hybrid Retrieval (BM25 + BGE-M3 + FAISS)**: Combines exact lexical term matches and semantic embeddings using Reciprocal Rank Fusion (RRF, $K=60$) and a Title Safety Net.
* **Honeypot Filter**: Applies strict chronological and capability contradiction checks, resulting in **0% honeypot infiltration** in the top 100.
* **Multi-Signal Blended Scorer**: Scores candidates across Technical (45%), Experience (25%), Behavioral/Availability (20%), and Location (10%) features, minus Red Flag penalties.
* **LLM Re-ranking**: Integrates DeepSeek-v4-flash re-ranking (40% score weight) and generates structured, evidence-grounded justifications.
* **NDCG Tier Normalizer**: Applies percentile boundaries to group candidates into NDCG quality bands.

---

## Technical Specs & Constraints Compliance
* **Ranking Step Runtime**: ~10 seconds (conforms to $\le$ 5 minutes spec).
* **Memory usage**: <1.5 GB RAM (conforms to $\le$ 16 GB spec).
* **CPU Only**: Conforms to CPU-only ranking requirement.
* **Zero Network during Ranking**: The ranking step runs entirely offline by reading precomputed embedding similarity indices and LLM scores.

---

## Setup & Execution

### 1. Installation
Install the required packages:
```bash
pip install -r sample/requirements.txt
```

### 2. Pre-computation (Offline, Network Allowed)
*Note: This step generates embeddings and calls the DeepSeek API. The pre-computed artifacts are already saved under `artifacts_bge/`.*
If you need to run it again, set your `DEEPSEEK_API_KEY` env var and run:
```bash
python precompute.py --candidates ./sample/candidates.jsonl --jd ./sample/job_description.txt --out ./artifacts_bge
```

### 3. Single-Command Reproduction (Stage 3 Verification)
To run the offline ranking step and reproduce the final output:
```bash
python rank.py --candidates ./sample/candidates.jsonl --artifacts ./artifacts_bge --out ./submission.csv
```

### 4. Sandbox Web Application
To run the Streamlit validation sandbox:
```bash
streamlit run app.py
```
