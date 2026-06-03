import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from src.config import CORE_SKILLS_MUST_HAVE, CORE_SKILLS_NICE_TO_HAVE
from src.precompute.embedder import GeminiEmbedder


def _extract_skills(jd_text: str, known_skills: List[str]) -> List[str]:
    lowered = jd_text.lower()
    return sorted({skill for skill in known_skills if skill.lower() in lowered})


def embed_jd(
    embedder: GeminiEmbedder,
    jd_text: str,
    output_path: Path,
) -> Dict[str, str]:
    skills = _extract_skills(jd_text, CORE_SKILLS_MUST_HAVE + CORE_SKILLS_NICE_TO_HAVE)
    skills_text = ", ".join(skills) if skills else ""

    vectors = embedder.embed_texts([jd_text, skills_text])
    payload = {
        "jd_profile_text": jd_text,
        "jd_skills_text": skills_text,
        "profile_embedding": vectors[0].tolist(),
        "skills_embedding": vectors[1].tolist(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": embedder._model_name,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return payload
