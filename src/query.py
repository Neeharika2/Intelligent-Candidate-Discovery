import json
import os
from openai import OpenAI


def _get_client(api_key: str = None) -> OpenAI:
    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    return OpenAI(api_key=key, base_url="https://api.deepseek.com/v1")


def extract_ideal_profile(jd_text: str, api_key: str = None) -> str:
    prompt = f"""You are a hiring expert. Given this job description, extract ONLY the positive attributes of the ideal candidate. Do NOT include company info, instructions, disclaimers, or what the company does NOT want. Focus on what makes someone an excellent fit.

Output a concise 10-15 sentence description of the ideal candidate's:
- Experience type and depth
- Technical skills and production experience
- Career trajectory and company types
- Location and availability
- Behavioral and cultural fit signals

Job Description:
{jd_text}

Output ONLY the ideal candidate description, no preamble or meta-text."""

    client = _get_client(api_key)
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500,
        timeout=60.0,
    )
    return response.choices[0].message.content.strip()


def build_rerank_prompt(ideal_profile: str, candidate_text: str) -> str:
    return f"""You are evaluating a candidate for a Senior AI Engineer role.

Ideal Candidate Profile:
{ideal_profile}

Candidate Profile:
{candidate_text}

Rate this candidate 1-5:
5 = Excellent fit (production ML/ranking/retrieval engineer at product company, strong behavioral signals)
4 = Strong fit (relevant ML experience with some gaps)
3 = Moderate fit (adjacent experience, could transition)
2 = Weak fit (mostly unrelated but some ML exposure)
1 = No fit (wrong domain, no ML production experience)

Output ONLY valid JSON: {{"score": <1-5>, "reasoning": "<1-2 sentences with specific evidence>"}}"""