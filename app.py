import streamlit as st
import pandas as pd
import json
import os
import tempfile

from src.loader import load_candidates, build_retrieval_corpus
from src.honeypot import detect_honeypot
from src.scoring import combine_scores, tier_normalize
from src.reasoning import generate_reasoning


st.set_page_config(page_title="Candidate Ranking — Redrob Hackathon", layout="wide")
st.title("🎯 Intelligent Candidate Discovery & Ranking")
st.caption("Redrob Hackathon — Senior AI Engineer Candidate Ranker")

st.sidebar.header("Input")

# File upload for sandbox requirement
uploaded_file = st.sidebar.file_uploader(
    "Upload candidates (JSON/JSONL)", type=["json", "jsonl"]
)
candidates_path = st.sidebar.text_input(
    "Or enter file path", value="./sample/sample_candidates.jsonl"
)

if st.sidebar.button("🚀 Rank Candidates"):
    candidates = None

    # Try uploaded file first, then path
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        if uploaded_file.name.endswith(".json"):
            data = json.loads(content)
            candidates = data if isinstance(data, list) else [data]
        else:  # .jsonl
            candidates = [json.loads(line) for line in content.strip().split("\n") if line.strip()]
        st.sidebar.success(f"Loaded {len(candidates)} candidates from upload")
    elif os.path.exists(candidates_path):
        candidates = load_candidates(candidates_path)
        st.sidebar.success(f"Loaded {len(candidates)} candidates from path")
    else:
        st.error(f"File not found: {candidates_path}. Upload a file or fix the path.")

    if candidates:
        progress = st.progress(0, text="Scoring candidates...")
        honeypots = {}
        results = []
        all_scores = []
        all_evidence = {}

        for i, c in enumerate(candidates):
            flagged, reason = detect_honeypot(c)
            if flagged:
                honeypots[c["candidate_id"]] = reason
                continue

            feature_score, evidence = combine_scores(c)
            all_scores.append(feature_score)
            all_evidence[c["candidate_id"]] = evidence
            results.append({
                "candidate_id": c["candidate_id"],
                "feature_score": feature_score,
                "title": c.get("profile", {}).get("current_title", ""),
                "company": c.get("profile", {}).get("current_company", ""),
                "industry": c.get("profile", {}).get("current_industry", ""),
                "years": c.get("profile", {}).get("years_of_experience", 0),
                "country": c.get("profile", {}).get("country", ""),
                "location": c.get("profile", {}).get("location", ""),
                "candidate": c,
            })
            progress.progress((i + 1) / len(candidates), text=f"Scored {i+1}/{len(candidates)}")

        # Normalize scores
        normalized = [tier_normalize(s, all_scores) for s in all_scores]
        for i, r in enumerate(results):
            r["final_score"] = round(normalized[i], 4)

        results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))

        # Generate proper reasoning
        for r in results[:100]:
            cid = r["candidate_id"]
            evidence = all_evidence.get(cid, {})
            r["reasoning"] = generate_reasoning(r["candidate"], evidence)

        progress.empty()

        # Display results
        col1, col2, col3 = st.columns(3)
        col1.metric("Candidates Scored", len(results))
        col2.metric("Honeypots Filtered", len(honeypots))
        col3.metric("Output", f"Top {min(100, len(results))}")

        st.subheader("📊 Top 100 Candidates")
        display_df = pd.DataFrame(results[:100])
        display_df["rank"] = range(1, len(display_df) + 1)
        st.dataframe(
            display_df[["rank", "candidate_id", "final_score", "title", "company",
                        "industry", "years", "country", "reasoning"]],
            use_container_width=True,
            height=500,
        )

        # Download CSV
        csv_df = display_df[["candidate_id", "rank", "final_score", "reasoning"]].rename(
            columns={"final_score": "score"}
        )
        st.download_button(
            "📥 Download Submission CSV",
            csv_df.to_csv(index=False),
            "submission.csv",
            "text/csv",
        )

        # Honeypot details
        if honeypots:
            with st.expander(f"🍯 Honeypots Detected ({len(honeypots)})"):
                for cid, reason in honeypots.items():
                    st.text(f"{cid}: {reason}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Sandbox demo** for the Redrob Hackathon.")
st.sidebar.markdown("For full pipeline with DeepSeek re-ranking:")
st.sidebar.code("python precompute.py --candidates ./sample/candidates.jsonl --jd ./sample/job_description.txt --out ./artifacts/\npython rank.py --candidates ./sample/candidates.jsonl --out ./submission.csv", language="bash")