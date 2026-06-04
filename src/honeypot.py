from src.config import CONSULTING_INDUSTRIES


def detect_honeypot(candidate: dict) -> tuple[bool, str]:
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    total_months = sum(h.get("duration_months", 0) for h in career)
    claimed_months = profile.get("years_of_experience", 0) * 12
    if claimed_months > 0 and total_months > claimed_months * 1.8 + 24:
        return True, f"career_months={total_months} vs claimed={claimed_months:.0f}"

    expert_zero = [
        s for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    ]
    if len(expert_zero) >= 3:
        return True, f"{len(expert_zero)} expert skills with zero duration"

    for edu in candidate.get("education", []):
        dur = edu.get("end_year", 0) - edu.get("start_year", 0)
        if dur < 0 or dur > 10:
            return True, f"education duration {dur}yr at {edu.get('institution', '?')}"

    assessments = signals.get("skill_assessment_scores", {})
    if assessments:
        for skill in skills:
            if skill.get("proficiency") == "expert" and skill.get("name") in assessments:
                if assessments[skill["name"]] < 30:
                    return True, f"expert claim contradicted by assessment ({skill['name']}={assessments[skill['name']]})"

    for h in career:
        if h.get("is_current") and h.get("duration_months") == 0:
            return True, f"current role with 0 months at {h.get('company', '?')}"

    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    total_endorsements = sum(s.get("endorsements", 0) for s in skills)
    if expert_count >= 5 and total_endorsements == 0:
        return True, f"{expert_count} expert skills, 0 endorsements"

    return False, ""