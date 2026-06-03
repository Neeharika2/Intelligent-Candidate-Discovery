import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jsonschema import Draft7Validator


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def _normalize_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    profile = candidate.get("profile", {})
    profile["headline"] = _normalize_text(profile.get("headline", ""))
    profile["summary"] = _normalize_text(profile.get("summary", ""))
    profile["current_title"] = _normalize_text(profile.get("current_title", ""))
    profile["current_company"] = _normalize_text(profile.get("current_company", ""))
    profile["current_industry"] = _normalize_text(profile.get("current_industry", ""))
    profile["location"] = _normalize_text(profile.get("location", ""))
    profile["country"] = _normalize_text(profile.get("country", ""))
    candidate["profile"] = profile

    for role in candidate.get("career_history", []):
        role["company"] = _normalize_text(role.get("company", ""))
        role["title"] = _normalize_text(role.get("title", ""))
        role["industry"] = _normalize_text(role.get("industry", ""))
        role["description"] = _normalize_text(role.get("description", ""))

    for skill in candidate.get("skills", []):
        skill["name"] = _normalize_text(skill.get("name", ""))

    return candidate


def _load_schema(schema_path: Path) -> Draft7Validator:
    with open(schema_path, "r", encoding="utf-8") as schema_file:
        schema = json.load(schema_file)
    return Draft7Validator(schema)


def load_candidates(
    jsonl_path: Path,
    schema_path: Path,
    max_records: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load and validate candidates from JSONL.

    Returns a tuple of (valid_candidates, invalid_records).
    """
    validator = _load_schema(schema_path)
    valid_candidates: List[Dict[str, Any]] = []
    invalid_records: List[Dict[str, Any]] = []

    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            if max_records is not None and len(valid_candidates) >= max_records:
                break
            raw = line.strip()
            if not raw:
                continue
            try:
                candidate = json.loads(raw)
            except json.JSONDecodeError as exc:
                invalid_records.append(
                    {"line": idx, "error": f"json_decode_error: {exc}"}
                )
                continue

            errors = sorted(validator.iter_errors(candidate), key=lambda e: e.path)
            if errors:
                invalid_records.append(
                    {
                        "line": idx,
                        "candidate_id": candidate.get("candidate_id"),
                        "error": "; ".join(err.message for err in errors),
                    }
                )
                continue

            valid_candidates.append(_normalize_candidate(candidate))

    return valid_candidates, invalid_records
