"""Step 6 - Prepare the job description and its requirements.

Two files describe a job:

    job/job_description.txt    the advert as free text (what HR writes)
    job/job_requirements.json  the same requirements as a structured list

Each structured requirement has:
    id           short name such as "R1"
    category     skill / experience / education / language / certification
    requirement  the requirement in words
    must_have    true  -> essential, a candidate without it is not qualified
                 false -> optional, only improves the candidate's ranking
    min_years    (experience only, optional) minimum number of years

HR stays in control: the review screen lets HR switch any requirement
between Must-Have and optional before candidates are compared.

Show the requirements :  python -m src.job_description
Review Must-Haves     :  python -m src.job_description --review
"""
import json
import logging
import os

from src.config import JOB_DESCRIPTION_PATH, JOB_REQUIREMENTS_PATH

logger = logging.getLogger(__name__)

CATEGORIES = ("skill", "experience", "education", "language", "certification")
CATEGORY_TITLES = {
    "skill": "Required skills",
    "experience": "Required experience",
    "education": "Education",
    "language": "Languages",
    "certification": "Certifications",
}


class JobDescriptionError(Exception):
    """Raised when a job file is missing or has invalid content."""


# ---------------------------------------------------------------------------
# 1. Loading and saving
# ---------------------------------------------------------------------------
def load_job_description(path=JOB_DESCRIPTION_PATH):
    """Return the job description text."""
    if not os.path.exists(path):
        raise JobDescriptionError(f"Job description file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise JobDescriptionError(f"Job description file is empty: {path}")
    return text


def validate_job(job):
    """Check the structure of a job dictionary. Raises JobDescriptionError.

    Checking early gives a clear message here instead of a confusing
    crash later in the matching step.
    """
    if not isinstance(job, dict) or not isinstance(job.get("requirements"), list):
        raise JobDescriptionError('Job file must contain a "requirements" list')
    if not job["requirements"]:
        raise JobDescriptionError("Job file has no requirements")

    seen_ids = set()
    for position, item in enumerate(job["requirements"], start=1):
        label = f"Requirement #{position}"
        if not isinstance(item, dict):
            raise JobDescriptionError(f"{label} must be an object")
        req_id = item.get("id")
        if not isinstance(req_id, str) or not req_id.strip():
            raise JobDescriptionError(f'{label} needs a text "id"')
        if req_id in seen_ids:
            raise JobDescriptionError(f'Duplicate requirement id "{req_id}"')
        seen_ids.add(req_id)
        if item.get("category") not in CATEGORIES:
            raise JobDescriptionError(
                f'{req_id}: "category" must be one of {", ".join(CATEGORIES)}')
        text = item.get("requirement")
        if not isinstance(text, str) or not text.strip():
            raise JobDescriptionError(f'{req_id}: "requirement" must be non-empty text')
        if not isinstance(item.get("must_have"), bool):
            raise JobDescriptionError(f'{req_id}: "must_have" must be true or false')
        years = item.get("min_years")
        if years is not None and (isinstance(years, bool)
                                  or not isinstance(years, (int, float)) or years < 0):
            raise JobDescriptionError(f'{req_id}: "min_years" must be a positive number')
    return job


def load_job(path=JOB_REQUIREMENTS_PATH):
    """Read and validate the structured requirements file."""
    if not os.path.exists(path):
        raise JobDescriptionError(f"Job requirements file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            job = json.load(f)
    except json.JSONDecodeError as e:
        raise JobDescriptionError(f"{path} is not valid JSON: {e}") from e
    return validate_job(job)


def save_job(job, path=JOB_REQUIREMENTS_PATH):
    """Validate and write the structured requirements file."""
    validate_job(job)
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# 2. Must-Have vs optional
# ---------------------------------------------------------------------------
def split_by_priority(job):
    """Return (must_have_requirements, optional_requirements)."""
    must = [r for r in job["requirements"] if r["must_have"]]
    optional = [r for r in job["requirements"] if not r["must_have"]]
    return must, optional


def _find(job, req_id):
    for requirement in job["requirements"]:
        if requirement["id"].lower() == req_id.lower():
            return requirement
    raise JobDescriptionError(f'No requirement with id "{req_id}"')


def toggle_must_have(job, req_ids):
    """Switch the given requirements between Must-Have and optional."""
    targets = [_find(job, req_id) for req_id in req_ids]  # fail before changing
    for requirement in targets:
        requirement["must_have"] = not requirement["must_have"]
    return job


def set_must_haves(job, req_ids):
    """Make exactly these requirements Must-Have and all others optional."""
    chosen = {_find(job, req_id)["id"] for req_id in req_ids}
    for requirement in job["requirements"]:
        requirement["must_have"] = requirement["id"] in chosen
    return job


# ---------------------------------------------------------------------------
# 3. Showing and reviewing
# ---------------------------------------------------------------------------
def format_job(job):
    """Return the requirements as readable text, grouped by category."""
    lines = [f"Job: {job.get('job_title', '(no title)')}"]
    for category in CATEGORIES:
        items = [r for r in job["requirements"] if r["category"] == category]
        if not items:
            continue
        lines.append(f"\n{CATEGORY_TITLES[category]}:")
        for r in items:
            flag = "MUST-HAVE" if r["must_have"] else "optional "
            lines.append(f"  {r['id']:<4} [{flag}] {r['requirement']}")
    must, optional = split_by_priority(job)
    lines.append(f"\n{len(must)} Must-Have, {len(optional)} optional")
    return "\n".join(lines)


def review_must_haves(job, path=JOB_REQUIREMENTS_PATH, input_func=input, output=print):
    """Let HR choose which requirements are Must-Have, then save.

    `input_func` and `output` exist so the tests can drive this screen
    without a real keyboard.
    """
    while True:
        output("\n" + format_job(job))
        output("\nType requirement ids to switch them (example: R5 R7),")
        output("'s' to save and exit, or 'q' to exit without saving.")
        answer = input_func("> ").strip()
        if answer.lower() == "q":
            output("No changes saved.")
            return False
        if answer.lower() == "s":
            if not split_by_priority(job)[0]:
                output("Warning: no Must-Have requirements are selected. "
                       "Every candidate will pass the Must-Have filter.")
            save_job(job, path)
            output(f"Saved to {path}")
            return True
        if not answer:
            continue
        try:
            toggle_must_have(job, answer.replace(",", " ").split())
        except JobDescriptionError as e:
            output(f"Not changed: {e}")


if __name__ == "__main__":
    import sys

    try:
        current_job = load_job()
        if "--review" in sys.argv:
            review_must_haves(current_job)
        else:
            print(f"Job description text: {len(load_job_description())} characters "
                  f"({JOB_DESCRIPTION_PATH})\n")
            print(format_job(current_job))
            print("\nTo change the Must-Have selection: "
                  "python -m src.job_description --review")
    except JobDescriptionError as e:
        print(f"Job description problem: {e}")
        sys.exit(1)
