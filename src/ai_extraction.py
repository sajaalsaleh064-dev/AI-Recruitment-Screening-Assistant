"""Step 7b - Use the LLM to turn free text into structured information.

Two separate extraction processes:

    A. Job description text -> required skills, experience, education,
                               languages, certifications
    B. CV text              -> candidate skills, work experience, education,
                               languages, certifications

The LLM only EXTRACTS. It does not score or judge anyone.

Protection against invented information ("hallucination"):
1. The prompt forbids guessing and asks for null / empty lists instead.
2. A JSON schema forces the structure of the answer.
3. The answer is cleaned and validated in Python (normalize_* functions).
4. Grounding check: every extracted item must really appear in the source
   text. Anything that does not is moved to an "unverified" list that HR
   can see, and is not used for matching.

Run on the sample job : python -m src.ai_extraction job
Run on stored CVs     : python -m src.ai_extraction cvs
"""
import logging
import re
from datetime import date

from src.config import MAX_LLM_INPUT_CHARS
from src.ollama_client import generate_json

logger = logging.getLogger(__name__)

JOB_CATEGORIES = {
    "required_skills": "skill",
    "required_experience": "experience",
    "education": "education",
    "languages": "language",
    "certifications": "certification",
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an information extraction tool used by an HR team.
You copy facts from a document into JSON. You never evaluate, score or judge.

Strict rules:
1. Use ONLY information written in the document. Never guess or add anything.
2. If something is not written in the document, use null or an empty list.
3. Copy names of skills, companies, degrees and certificates as they are written.
4. Do not turn a wish or a plan ("interested in learning Docker") into a skill.
5. Answer with one JSON object only. No explanations."""

CV_PROMPT = """Extract the candidate's information from the CV below.

Return a JSON object with exactly these keys:
- "name": the candidate's full name, or null.
- "skills": list of technical and professional skills, one per item
  (for example "Python", not "Python, SQL").
- "work_experience": list of jobs, each with "job_title", "company",
  "start_date", "end_date" (as written, for example "Jun 2024" or "Present")
  and "description" (one short sentence copied or shortened from the CV).
- "stated_years_experience": a number ONLY if the CV itself states a total
  such as "3 years of experience"; otherwise null. Do not calculate it.
- "education": list of entries, each with "degree", "field", "institution", "year".
- "languages": list of spoken languages, each with "language" and "level"
  (level is null if not written). Programming languages are NOT languages.
- "certifications": list of certificate names. Degrees are not certifications.

CV:
\"\"\"
{text}
\"\"\""""

JOB_PROMPT = """Extract the requirements from the job description below.

Return a JSON object with exactly these keys:
- "job_title": the job title, or null.
- "required_skills", "required_experience", "education", "languages",
  "certifications": each is a list of requirements.

Every requirement is an object with:
- "requirement": one requirement in a short phrase copied from the text.
  Split combined lines into separate requirements only when they name
  clearly different things.
- "must_have": true if the text presents it as required / essential /
  must-have; false if it is presented as nice-to-have, preferred, a plus
  or optional. If the text does not say, use true.
- "min_years": for experience requirements, the minimum number of years
  as a number if stated; otherwise null.

Put each requirement in ONE list only: spoken languages go in "languages",
certificates in "certifications", degrees in "education", years of experience
in "required_experience", everything else in "required_skills".
Do not list responsibilities or company information as requirements.

Job description:
\"\"\"
{text}
\"\"\""""

# ---------------------------------------------------------------------------
# JSON schemas (the structure the model is forced to follow)
# ---------------------------------------------------------------------------
_NULLABLE_STRING = {"type": ["string", "null"]}
_NULLABLE_NUMBER = {"type": ["number", "null"]}

CV_SCHEMA = {
    "type": "object",
    "properties": {
        "name": _NULLABLE_STRING,
        "skills": {"type": "array", "items": {"type": "string"}},
        "work_experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "job_title": _NULLABLE_STRING,
                    "company": _NULLABLE_STRING,
                    "start_date": _NULLABLE_STRING,
                    "end_date": _NULLABLE_STRING,
                    "description": _NULLABLE_STRING,
                },
                "required": ["job_title", "company", "start_date", "end_date", "description"],
            },
        },
        "stated_years_experience": _NULLABLE_NUMBER,
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "degree": _NULLABLE_STRING,
                    "field": _NULLABLE_STRING,
                    "institution": _NULLABLE_STRING,
                    "year": _NULLABLE_STRING,
                },
                "required": ["degree", "field", "institution", "year"],
            },
        },
        "languages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"language": {"type": "string"}, "level": _NULLABLE_STRING},
                "required": ["language", "level"],
            },
        },
        "certifications": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "skills", "work_experience", "stated_years_experience",
                 "education", "languages", "certifications"],
}

_REQUIREMENT_LIST = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "requirement": {"type": "string"},
            "must_have": {"type": "boolean"},
            "min_years": _NULLABLE_NUMBER,
        },
        "required": ["requirement", "must_have", "min_years"],
    },
}

JOB_SCHEMA = {
    "type": "object",
    "properties": {"job_title": _NULLABLE_STRING,
                   **{key: _REQUIREMENT_LIST for key in JOB_CATEGORIES}},
    "required": ["job_title", *JOB_CATEGORIES],
}


# ---------------------------------------------------------------------------
# Small helpers for cleaning values
# ---------------------------------------------------------------------------
_EMPTY_WORDS = {"", "n/a", "na", "none", "null", "unknown", "not specified",
                "not mentioned", "not provided", "-"}


def _text(value):
    """Return a clean string, or None for missing / placeholder values."""
    if value is None or isinstance(value, (dict, list, bool)):
        return None
    value = re.sub(r"\s+", " ", str(value)).strip()
    return None if value.lower() in _EMPTY_WORDS else value


def _number(value):
    """Return a non-negative number, or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value if value >= 0 else None
    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value)
        return float(match.group()) if match else None
    return None


def _list(value):
    """Always return a list (models sometimes return null or one string)."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _unique_strings(values):
    """Clean a list of strings and drop duplicates (ignoring upper/lower case)."""
    seen, result = set(), []
    for value in _list(values):
        if isinstance(value, dict):  # for example {"name": "Python"}
            value = next((v for v in value.values() if isinstance(v, str)), None)
        value = _text(value)
        if value and value.lower() not in seen:
            seen.add(value.lower())
            result.append(value)
    return result


def _normalize_for_search(text):
    return re.sub(r"\s+", " ", text.lower())


def appears_in_text(phrase, source_text):
    """Grounding check: is this phrase really written in the source text?

    True if the whole phrase is found, or if every word of it is found.
    Upper/lower case and extra spaces are ignored.
    """
    if not phrase or not source_text:
        return False
    source = _normalize_for_search(source_text)
    target = _normalize_for_search(phrase)
    if target in source:
        return True
    words = re.findall(r"[a-z0-9+#.]+", target)
    words = [w.strip(".") for w in words if len(w.strip(".")) > 1]
    return bool(words) and all(word in source for word in words)


# ---------------------------------------------------------------------------
# Experience length, calculated in Python (LLMs are unreliable at arithmetic)
# ---------------------------------------------------------------------------
_MONTHS = {name: number for number, names in enumerate(
    [("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"),
     ("may",), ("jun", "june"), ("jul", "july"), ("aug", "august"),
     ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
     ("dec", "december")], start=1) for name in names}
_PRESENT_WORDS = ("present", "current", "now", "today", "ongoing", "to date")


def parse_month(value, today=None, is_end=False):
    """Turn "Jun 2024", "06/2024", "2024" or "Present" into (year, month).

    Returns None when the text holds no recognisable date.
    """
    today = today or date.today()
    if not value:
        return None
    text = str(value).lower().strip()
    # \b = whole words only, so "unknown" is not read as "now".
    if re.search(rf"\b(?:{'|'.join(_PRESENT_WORDS)})\b", text):
        return (today.year, today.month)
    year_match = re.search(r"(19|20)\d{2}", text)
    if not year_match:
        return None
    year = int(year_match.group())
    for word in re.findall(r"[a-z]+", text):
        if word in _MONTHS:
            return (year, _MONTHS[word])
    number_match = re.search(r"\b(0?[1-9]|1[0-2])\s*[/.-]\s*(?:19|20)\d{2}", text)
    if number_match:
        return (year, int(number_match.group(1)))
    # Only a year is known: assume January for a start, December for an end.
    return (year, 12 if is_end else 1)


def calculate_years_experience(work_experience, today=None):
    """Add up the length of all jobs, counting overlapping months once.

    Returns years rounded to one decimal, or None if no job has usable dates.
    """
    today = today or date.today()
    months = set()
    for job in work_experience:
        start = parse_month(job.get("start_date"), today)
        end = parse_month(job.get("end_date"), today, is_end=True)
        if not start or not end:
            continue
        first, last = start[0] * 12 + start[1], end[0] * 12 + end[1]
        if last < first or last > today.year * 12 + today.month:
            continue  # impossible dates: ignore instead of guessing
        months.update(range(first, last + 1))
    return round(len(months) / 12, 1) if months else None


def find_stated_years(cv_text):
    """Find a total the CV states itself, such as "4 years of experience".

    Returns the number, or None if the CV does not state one.
    """
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\b(?:\s+of)?(?:\s+[\w-]+){0,3}?\s+experience",
        cv_text.lower())
    return float(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# A. CV extraction
# ---------------------------------------------------------------------------
def normalize_cv_info(raw, cv_text, today=None):
    """Clean and validate the model's CV answer.

    - Missing keys become null / empty lists, so later code never crashes.
    - Wrong types are repaired where possible.
    - Items that do not appear in the CV text go to "unverified".
    """
    raw = raw if isinstance(raw, dict) else {}
    unverified = []

    def keep_if_grounded(category, label, *evidence):
        """True if any of the evidence values appears in the CV text."""
        if any(appears_in_text(value, cv_text) for value in evidence if value):
            return True
        unverified.append({"category": category, "value": label})
        logger.warning("Not found in CV text, moved to unverified: %s = %r",
                       category, label)
        return False

    name = _text(raw.get("name"))
    if name and not appears_in_text(name, cv_text):
        unverified.append({"category": "name", "value": name})
        name = None

    skills = [s for s in _unique_strings(raw.get("skills"))
              if keep_if_grounded("skill", s, s)]

    certifications = [c for c in _unique_strings(raw.get("certifications"))
                      if keep_if_grounded("certification", c, c)]

    work_experience = []
    for item in _list(raw.get("work_experience")):
        if not isinstance(item, dict):
            continue
        job = {key: _text(item.get(key)) for key in
               ("job_title", "company", "start_date", "end_date", "description")}
        if not job["job_title"] and not job["company"]:
            continue
        label = " at ".join(v for v in (job["job_title"], job["company"]) if v)
        if keep_if_grounded("work_experience", label, job["company"], job["job_title"]):
            work_experience.append(job)

    education = []
    for item in _list(raw.get("education")):
        if not isinstance(item, dict):
            continue
        entry = {key: _text(item.get(key)) for key in
                 ("degree", "field", "institution", "year")}
        if not entry["degree"] and not entry["institution"] and not entry["field"]:
            continue
        label = ", ".join(v for v in (entry["degree"], entry["field"],
                                      entry["institution"]) if v)
        if keep_if_grounded("education", label, entry["degree"],
                            entry["institution"], entry["field"]):
            education.append(entry)

    languages, seen = [], set()
    for item in _list(raw.get("languages")):
        if isinstance(item, str):
            item = {"language": item, "level": None}
        if not isinstance(item, dict):
            continue
        language = _text(item.get("language"))
        if not language or language.lower() in seen:
            continue
        seen.add(language.lower())
        if keep_if_grounded("language", language, language):
            languages.append({"language": language, "level": _text(item.get("level"))})

    stated_years = _number(raw.get("stated_years_experience"))
    if stated_years is None:
        # The model often leaves this empty even when the CV says it, so
        # look for a sentence such as "4 years of experience" ourselves.
        stated_years = find_stated_years(cv_text)
    # A stated total is only trusted if that number is written in the CV.
    if stated_years is not None and not re.search(
            rf"(?<![\d.]){re.escape(f'{stated_years:g}')}\+?\s*(?:years|yrs|year)",
            cv_text.lower()):
        unverified.append({"category": "stated_years_experience", "value": stated_years})
        stated_years = None

    return {
        "name": name,
        "skills": skills,
        "work_experience": work_experience,
        "stated_years_experience": stated_years,
        "calculated_years_experience": calculate_years_experience(work_experience, today),
        "education": education,
        "languages": languages,
        "certifications": certifications,
        "unverified": unverified,
    }


def _shorten(text):
    if len(text) > MAX_LLM_INPUT_CHARS:
        logger.warning("Text is %d characters; only the first %d are analysed",
                       len(text), MAX_LLM_INPUT_CHARS)
        return text[:MAX_LLM_INPUT_CHARS]
    return text


def extract_cv_information(cv_text, llm=generate_json, today=None):
    """Extract structured candidate information from CV text.

    `llm` is the function that calls the model; tests replace it with a
    fake so they run without Ollama.
    """
    if not cv_text or not cv_text.strip():
        raise ValueError("CV text is empty; nothing to extract")
    raw = llm(SYSTEM_PROMPT, CV_PROMPT.format(text=_shorten(cv_text)), CV_SCHEMA)
    return normalize_cv_info(raw, cv_text, today)


# ---------------------------------------------------------------------------
# B. Job description extraction
# ---------------------------------------------------------------------------
def normalize_job_info(raw, job_text):
    """Clean and validate the model's job description answer."""
    raw = raw if isinstance(raw, dict) else {}
    result = {"job_title": _text(raw.get("job_title")), "unverified": []}
    # Models sometimes repeat a language or certificate under skills too.
    # Skills are handled last, so the copy in the specific list is the one kept.
    seen = set()
    order = sorted(JOB_CATEGORIES.items(), key=lambda pair: pair[1] == "skill")
    for key, category in order:
        cleaned = []
        for item in _list(raw.get(key)):
            if isinstance(item, str):
                item = {"requirement": item}
            if not isinstance(item, dict):
                continue
            requirement = _text(item.get("requirement"))
            if not requirement or requirement.lower() in seen:
                continue
            seen.add(requirement.lower())
            if not appears_in_text(requirement, job_text):
                result["unverified"].append({"category": category, "value": requirement})
                logger.warning("Not found in job text, moved to unverified: %r", requirement)
                continue
            must_have = item.get("must_have")
            entry = {
                "requirement": requirement,
                # If the model gave no clear answer, treat it as Must-Have so
                # HR sees it and can relax it in the review step.
                "must_have": must_have if isinstance(must_have, bool) else True,
            }
            years = _number(item.get("min_years")) if category == "experience" else None
            if years is not None:
                entry["min_years"] = years
            cleaned.append(entry)
        result[key] = cleaned
    return result


def extract_job_requirements(job_text, llm=generate_json):
    """Extract structured requirements from job description text."""
    if not job_text or not job_text.strip():
        raise ValueError("Job description text is empty; nothing to extract")
    raw = llm(SYSTEM_PROMPT, JOB_PROMPT.format(text=_shorten(job_text)), JOB_SCHEMA)
    return normalize_job_info(raw, job_text)


def job_from_extraction(job_info):
    """Convert extracted requirements into the Step 6 job format (R1, R2...),
    so HR can review them with: python -m src.job_description --review"""
    requirements = []
    for key, category in JOB_CATEGORIES.items():
        for item in job_info.get(key, []):
            requirement = {
                "id": f"R{len(requirements) + 1}",
                "category": category,
                "requirement": item["requirement"],
                "must_have": item["must_have"],
            }
            if "min_years" in item:
                requirement["min_years"] = item["min_years"]
            requirements.append(requirement)
    return {"job_title": job_info.get("job_title") or "Untitled job",
            "requirements": requirements}


# ---------------------------------------------------------------------------
# Working on the stored CV records (from Step 5)
# ---------------------------------------------------------------------------
def analyse_stored_cvs(force=False, llm=generate_json):
    """Add "cv_info" to every stored CV record that does not have it yet."""
    from src.ollama_client import OllamaError
    from src.text_extractor import load_cv_records, save_cv_record

    records = load_cv_records()
    for record in records:
        if record["extraction"]["status"] != "ok":
            logger.info("%s: no readable text, left for manual review",
                        record["original_filename"])
            continue
        if record.get("cv_info") and not force:
            continue
        logger.info("Analysing %s with the LLM...", record["original_filename"])
        try:
            record["cv_info"] = extract_cv_information(record["cv_text"], llm=llm)
        except OllamaError as e:
            record["cv_info_error"] = str(e)
            logger.error("%s: %s", record["original_filename"], e)
            continue
        record.pop("cv_info_error", None)
        save_cv_record(record)
    return records


def format_cv_info(info):
    """Readable summary of one candidate's extracted information."""
    def years(value):
        return "not available" if value is None else f"{value:g}"

    lines = [
        f"Name          : {info['name'] or 'not found'}",
        f"Skills        : {', '.join(info['skills']) or 'none found'}",
        f"Years (stated): {years(info['stated_years_experience'])}",
        f"Years (dates) : {years(info['calculated_years_experience'])}",
        "Experience    :",
    ]
    lines += [f"  - {j['job_title']} | {j['company']} | {j['start_date']} - {j['end_date']}"
              for j in info["work_experience"]] or ["  none found"]
    lines.append("Education     :")
    lines += [f"  - {e['degree']} | {e['field']} | {e['institution']} | {e['year']}"
              for e in info["education"]] or ["  none found"]
    lines.append("Languages     : " + (", ".join(
        f"{l['language']} ({l['level']})" if l["level"] else l["language"]
        for l in info["languages"]) or "none found"))
    lines.append("Certifications: " + (", ".join(info["certifications"]) or "none found"))
    if info["unverified"]:
        lines.append("Unverified (not found in the CV text, not used):")
        lines += [f"  - {u['category']}: {u['value']}" for u in info["unverified"]]
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    import sys

    from src.job_description import (JobDescriptionError, format_job,
                                     load_job_description, save_job)
    from src.ollama_client import OllamaError, check_ollama

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        check_ollama()
        if mode == "job":
            extracted = extract_job_requirements(load_job_description())
            job = job_from_extraction(extracted)
            print(format_job(job))
            if extracted["unverified"]:
                print("\nUnverified (not used):",
                      json.dumps(extracted["unverified"], indent=2))
            if "--save" in sys.argv:
                save_job(job)
                print("\nSaved. Review it with: python -m src.job_description --review")
            else:
                print("\nNot saved. Add --save to replace job/job_requirements.json "
                      "with this result.")
        elif mode == "cvs":
            for cv_record in analyse_stored_cvs(force="--force" in sys.argv):
                print(f"\n=== {cv_record['original_filename']} ===")
                if cv_record.get("cv_info"):
                    print(format_cv_info(cv_record["cv_info"]))
                else:
                    print(cv_record.get("cv_info_error")
                          or "No readable text: needs manual review by HR")
        else:
            print("Usage: python -m src.ai_extraction job [--save]\n"
                  "       python -m src.ai_extraction cvs [--force]")
    except (OllamaError, JobDescriptionError) as e:
        print(f"\nProblem: {e}")
        sys.exit(1)
