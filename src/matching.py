"""Step 8 - Compare one candidate with every job requirement.

For each requirement the candidate gets one of three results:

    Meets the Requirement
    Partially Meets the Requirement
    Does Not Meet the Requirement

Two techniques are combined:

1. Rules (exact, easy to explain)
   - a skill named in the requirement is in the candidate's skill list
   - years of relevant experience compared with the minimum as numbers
   - degree level (diploma < bachelor < master < PhD)
   - language name and level

2. Semantic similarity (meaning, not exact words)
   The requirement and the candidate's information are turned into
   vectors ("embeddings") with sentence-transformers/all-MiniLM-L6-v2.
   Cosine similarity between two vectors says how close their meanings
   are: about 1.0 = same meaning, about 0 = unrelated. This is how
   "GitHub" can match "Git for version control" without shared wording.

Every result carries evidence: lines copied directly from the CV. A result
without evidence is always "Does Not Meet" with "No evidence found".
Nothing here rejects anyone: the output is information for HR to review.

Run on the stored CVs:  python -m src.matching
"""
import logging
import re
from datetime import date

import numpy as np

from src.ai_extraction import calculate_years_experience
from src.config import (EMBEDDING_MODEL, SIMILARITY_MEETS, SIMILARITY_PARTIAL,
                        SIMILARITY_RELEVANT)

logger = logging.getLogger(__name__)

MEETS = "Meets the Requirement"
PARTIAL = "Partially Meets the Requirement"
NOT_MET = "Does Not Meet the Requirement"
NO_EVIDENCE = "No evidence found"

# A line that only expresses a wish is not evidence of a skill.
INTENT_PATTERN = re.compile(
    r"\b(interested in|interest in|would like to|want to|wish to|hope to|plan to|"
    r"planning to|eager to|willing to learn|keen to|currently learning|"
    r"in the future|aspire|aspiring)\b", re.IGNORECASE)
# Words in the CV that mark a skill as only basic.
BASIC_WORDS = r"(?:basic|basics|beginner|elementary|familiar|learning|introductory)"

DEGREE_LEVELS = [  # (level number, name, pattern), highest level first
    (4, "PhD", r"\b(ph\.?\s?d|doctorate|doctoral)\b"),
    (3, "Master's", r"\b(master'?s?|m\.?\s?sc|msc|m\.?s\.?c?|mba|m\.?eng|m\.?a)\b"),
    (2, "Bachelor's", r"\b(bachelor'?s?|b\.?\s?sc|bsc|b\.?s|b\.?eng|b\.?a|b\.?tech|undergraduate degree)\b"),
    (1, "Diploma", r"\b(diploma|associate'?s?( degree)?|certificate)\b"),
]
LANGUAGE_LEVELS = [  # (level number, pattern), highest first
    (3, r"\b(native|mother tongue|fluent|fluency|professional|proficient|proficiency|"
        r"advanced|bilingual|excellent|c1|c2)\b"),
    (2, r"\b(intermediate|upper[- ]intermediate|good|conversational|working knowledge|b1|b2)\b"),
    (1, r"\b(basic|beginner|elementary|limited|a1|a2)\b"),
]


# ---------------------------------------------------------------------------
# 1. Embeddings and cosine similarity
# ---------------------------------------------------------------------------
class Embedder:
    """Turns text into vectors with a Sentence Transformers model.

    The model is loaded only when first needed (it takes a while), and
    every text is remembered so it is never embedded twice.
    """

    def __init__(self, model_name=EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
        self._cache = {}

    def encode(self, texts):
        """Return one vector (384 numbers for all-MiniLM-L6-v2) per text."""
        texts = list(texts)
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            if self._model is None:
                # Imported here because loading PyTorch is slow.
                from sentence_transformers import SentenceTransformer
                logger.info("Loading embedding model %s ...", self.model_name)
                self._model = SentenceTransformer(self.model_name)
            vectors = self._model.encode(missing, show_progress_bar=False)
            for text, vector in zip(missing, vectors):
                self._cache[text] = np.asarray(vector, dtype=float)
        return [self._cache[t] for t in texts]


_default_embedder = None


def get_embedder():
    """Return one shared Embedder so the model is loaded only once."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = Embedder()
    return _default_embedder


def cosine_similarity(a, b):
    """Cosine of the angle between two vectors.

        similarity = (a . b) / (|a| * |b|)

    1 means same direction (same meaning), 0 means unrelated.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def best_match(embedder, query, candidates):
    """Return (similarity, candidate text) for the candidate closest in
    meaning to the query, or (0.0, None) when there are no candidates."""
    candidates = [c for c in candidates if c]
    if not candidates:
        return 0.0, None
    vectors = embedder.encode([query] + candidates)
    scores = [cosine_similarity(vectors[0], v) for v in vectors[1:]]
    index = int(np.argmax(scores))
    return scores[index], candidates[index]


# ---------------------------------------------------------------------------
# 2. Evidence: lines copied from the CV
# ---------------------------------------------------------------------------
def cv_lines(cv_text):
    """Split the CV text into clean lines."""
    lines = []
    for line in (cv_text or "").split("\n"):
        line = line.strip().lstrip("-").strip()
        if line:
            lines.append(line)
    return lines


def is_intent_only(line):
    """True for lines such as "Interested in learning Python"."""
    return bool(INTENT_PATTERN.search(line))


def _contains_term(text, term):
    """Whole-word search that also works for "C++", "C#" and plurals."""
    term = term.lower().strip()
    if not term:
        return False
    if len(term) > 3 and term.endswith("s"):
        term = term[:-1]
    pattern = r"(?<![a-z0-9])" + re.escape(term) + r"s?(?![a-z0-9+#])"
    return re.search(pattern, text.lower()) is not None


def find_evidence(lines, phrases, limit=3):
    """Return CV lines that mention any of the phrases (wish lines excluded).

    If a phrase is not found word for word, the line sharing most of its
    words is used, provided at least half of the words are there.
    """
    evidence = []
    usable = [line for line in lines if not is_intent_only(line)]
    for phrase in phrases:
        if not phrase:
            continue
        found = [line for line in usable if _contains_term(line, phrase)]
        if not found:
            words = [w for w in re.findall(r"[a-z0-9+#]+", phrase.lower()) if len(w) > 2]
            scored = [(sum(w in line.lower() for w in words), line) for line in usable]
            best = max(scored, default=(0, None))
            if words and best[0] * 2 >= len(words) and best[0] > 0:
                found = [best[1]]
        for line in found[:1]:
            if line not in evidence:
                evidence.append(line)
    return evidence[:limit]


# ---------------------------------------------------------------------------
# 3. One result
# ---------------------------------------------------------------------------
def _result(requirement, status, explanation, evidence=(), similarity=None, method="rule"):
    """Build one result. Enforces: no evidence -> cannot meet the requirement."""
    evidence = list(evidence)
    if status != NOT_MET and not evidence:
        status = NOT_MET
        explanation += " No supporting line could be found in the CV, so it is not counted."
    if status == NOT_MET:
        evidence = []
    return {
        "requirement_id": requirement["id"],
        "category": requirement["category"],
        "requirement": requirement["requirement"],
        "must_have": requirement["must_have"],
        "status": status,
        "evidence": evidence if evidence else [NO_EVIDENCE],
        "similarity": None if similarity is None else round(similarity, 2),
        "method": method,
        "explanation": explanation,
    }


def _status_from_similarity(score):
    if score >= SIMILARITY_MEETS:
        return MEETS
    if score >= SIMILARITY_PARTIAL:
        return PARTIAL
    return NOT_MET


# ---------------------------------------------------------------------------
# 4. Matching per category
# ---------------------------------------------------------------------------
def _semantic_units(profile):
    """Texts a requirement may be compared with: the extracted skills and
    the CV's sentences (not contact lines, not wishes)."""
    sentences = [line for line in profile["lines"]
                 if len(line.split()) >= 3 and "@" not in line and not is_intent_only(line)]
    return list(dict.fromkeys(profile["info"]["skills"] + sentences))


def match_skill(requirement, profile, embedder):
    text = requirement["requirement"]
    info, lines = profile["info"], profile["lines"]

    # Rule: a skill from the candidate's list is named in the requirement.
    named = [s for s in info["skills"] if _contains_term(text, s)]
    evidence = find_evidence(lines, named)
    if named and evidence:
        only_basic = all(
            re.search(re.escape(s.lower()) + r"\s*\(\s*" + BASIC_WORDS,
                      profile["text"].lower()) for s in named)
        if only_basic:
            return _result(requirement, PARTIAL,
                           f"The CV lists {', '.join(named)}, but describes it as basic.",
                           evidence)
        return _result(requirement, MEETS,
                       f"The CV lists {', '.join(named)}, named in this requirement.",
                       evidence)

    # Semantic: closest skill or sentence by meaning.
    score, closest = best_match(embedder, text, _semantic_units(profile))
    status = _status_from_similarity(score)
    if status == NOT_MET:
        return _result(requirement, NOT_MET,
                       f"{NO_EVIDENCE}. Nothing in the CV is close in meaning "
                       f"(highest similarity {score:.2f}).", similarity=score,
                       method="rule+semantic")
    evidence = find_evidence(lines, [closest])
    strength = ("closely matches" if status == MEETS
                else "is only loosely related to")
    return _result(requirement, status,
                   f'No exact keyword match, but "{closest}" {strength} this '
                   f"requirement in meaning (similarity {score:.2f}).",
                   evidence, score, "semantic")


def experience_topic(requirement_text):
    """"At least 2 years of experience in Python development" -> "Python development"."""
    topic = re.sub(
        r"^\s*(?:at least|minimum(?: of)?|min\.?|over|more than)?\s*\d+(?:\.\d+)?\s*\+?\s*"
        r"(?:years?|yrs?)(?:\s+of)?(?:\s+[\w-]+){0,2}?\s+experience(?:\s+(?:in|with|as|of))?\s*",
        "", requirement_text, flags=re.IGNORECASE).strip()
    return topic or requirement_text


def match_experience(requirement, profile, embedder, today=None):
    info, lines = profile["info"], profile["lines"]
    jobs = info["work_experience"]
    if not jobs:
        return _result(requirement, NOT_MET,
                       f"{NO_EVIDENCE}. The CV lists no work experience.")

    # Which jobs are related to the field the requirement asks for?
    topic = experience_topic(requirement["requirement"])
    scored = []
    for job in jobs:
        title = job["job_title"] or ""
        texts = [title, f"{title}: {job['description']}" if job["description"] else None]
        score, _ = best_match(embedder, topic, texts)
        scored.append((score, job))
    relevant = [job for score, job in scored if score >= SIMILARITY_RELEVANT]
    top_score, top_job = max(scored, key=lambda pair: pair[0])

    if not relevant:
        return _result(
            requirement, NOT_MET,
            f"{NO_EVIDENCE} of related experience. The CV lists {len(jobs)} job(s), "
            f'but none is related to "{topic}" (closest: {top_job["job_title"]}, '
            f"similarity {top_score:.2f}).", similarity=top_score, method="rule+semantic")

    evidence = find_evidence(lines, [j["company"] or j["job_title"] for j in relevant])
    titles = ", ".join(j["job_title"] or "?" for j in relevant)
    minimum = requirement.get("min_years")
    if minimum is None:
        status = _status_from_similarity(top_score)
        status = PARTIAL if status == NOT_MET else status
        return _result(requirement, status,
                       f"Related experience found: {titles} (similarity {top_score:.2f}).",
                       evidence, top_score, "rule+semantic")

    years = calculate_years_experience(relevant, today)
    source = "calculated from the job dates"
    if years is None and info["stated_years_experience"] is not None:
        years, source = info["stated_years_experience"], "stated in the CV"
    if years is None:
        return _result(requirement, PARTIAL,
                       f"Related experience found ({titles}), but the CV gives no dates, "
                       f"so the {minimum:g} year minimum cannot be checked.",
                       evidence, top_score, "rule+semantic")
    detail = f"{years:g} year(s) of related experience ({titles}), {source}"
    if years >= minimum:
        return _result(requirement, MEETS,
                       f"{detail}; the minimum is {minimum:g}.",
                       evidence, top_score, "rule+semantic")
    return _result(requirement, PARTIAL,
                   f"{detail}; this is below the minimum of {minimum:g}.",
                   evidence, top_score, "rule+semantic")


def degree_level(text):
    """Return (level number, level name) for the highest degree named in the text."""
    for level, name, pattern in DEGREE_LEVELS:
        if re.search(pattern, (text or "").lower()):
            return level, name
    return 0, None


def match_education(requirement, profile, embedder):
    text = requirement["requirement"]
    info, lines = profile["info"], profile["lines"]
    if not info["education"]:
        return _result(requirement, NOT_MET,
                       f"{NO_EVIDENCE}. The CV lists no education.")

    required_level, required_name = degree_level(text)
    best = None
    for entry in info["education"]:
        described = " in ".join(v for v in (entry["degree"], entry["field"]) if v)
        level, level_name = degree_level(entry["degree"])
        score, _ = best_match(embedder, text, [described])
        field_ok = bool(entry["field"] and _contains_term(text, entry["field"])) \
            or score >= SIMILARITY_MEETS
        level_ok = required_level == 0 or level >= required_level
        rank = (level_ok + field_ok, score)
        if best is None or rank > best[0]:
            best = (rank, entry, described, level_name, level_ok, field_ok, score)

    _, entry, described, level_name, level_ok, field_ok, score = best
    evidence = find_evidence(lines, [entry["institution"], entry["degree"], entry["field"]],
                             limit=2)
    if level_ok and field_ok:
        return _result(requirement, MEETS,
                       f'"{described}" has the required level and a matching field '
                       f"(similarity {score:.2f}).", evidence, score, "rule+semantic")
    if level_ok:
        return _result(requirement, PARTIAL,
                       f'"{described}" has the required level, but the field does not '
                       f"clearly match (similarity {score:.2f}). HR should judge whether "
                       "it is a related field.", evidence, score, "rule+semantic")
    if field_ok:
        return _result(requirement, PARTIAL,
                       f'"{described}" is in a matching field, but the level '
                       f"({level_name or 'not stated'}) is below the required "
                       f"{required_name}.", evidence, score, "rule+semantic")
    return _result(requirement, NOT_MET,
                   f'{NO_EVIDENCE} of the required education. The CV lists "{described}", '
                   f"which has neither the required level nor a matching field "
                   f"(similarity {score:.2f}).", similarity=score, method="rule+semantic")


def language_level(text):
    for level, pattern in LANGUAGE_LEVELS:
        if re.search(pattern, (text or "").lower()):
            return level
    return 0  # not stated


def match_language(requirement, profile, embedder=None):
    text = requirement["requirement"]
    info, lines = profile["info"], profile["lines"]
    spoken = [l for l in info["languages"] if _contains_term(text, l["language"])]
    if not spoken:
        return _result(requirement, NOT_MET,
                       f"{NO_EVIDENCE}. The CV does not list this language.")
    language = spoken[0]
    evidence = find_evidence(lines, [language["language"]], limit=1)
    required = language_level(text)
    actual = language_level(language["level"])
    shown = language["level"] or "level not stated"
    if required == 0 or actual >= required:
        return _result(requirement, MEETS,
                       f"The CV lists {language['language']} ({shown}).", evidence)
    return _result(requirement, PARTIAL,
                   f"The CV lists {language['language']} ({shown}), which is below "
                   "the level the requirement asks for.", evidence)


def match_certification(requirement, profile, embedder):
    text = requirement["requirement"]
    info, lines = profile["info"], profile["lines"]
    certificates = info["certifications"]
    if not certificates:
        return _result(requirement, NOT_MET,
                       f"{NO_EVIDENCE}. The CV lists no certifications.")

    def short(name):  # "AWS Cloud Practitioner (2023)" -> "AWS Cloud Practitioner"
        return re.sub(r"\s*\(.*?\)\s*$", "", name).strip()

    named = [c for c in certificates if _contains_term(text, short(c))]
    if named:
        return _result(requirement, MEETS,
                       f"The CV lists {named[0]}, named in this requirement.",
                       find_evidence(lines, [short(named[0])]))

    score, closest = best_match(embedder, text, certificates)
    status = _status_from_similarity(score)
    if status == NOT_MET:
        return _result(requirement, NOT_MET,
                       f"{NO_EVIDENCE}. The CV's certifications are not related "
                       f'(closest: "{closest}", similarity {score:.2f}).',
                       similarity=score, method="rule+semantic")
    return _result(requirement, status,
                   f'"{closest}" is similar in meaning (similarity {score:.2f}). '
                   "HR should confirm it is equivalent.",
                   find_evidence(lines, [short(closest)]), score, "semantic")


MATCHERS = {
    "skill": match_skill,
    "experience": match_experience,
    "education": match_education,
    "language": match_language,
    "certification": match_certification,
}


# ---------------------------------------------------------------------------
# 5. One candidate against the whole job
# ---------------------------------------------------------------------------
def match_candidate(job, record, embedder=None, today=None):
    """Compare one stored CV record with every requirement of the job."""
    candidate = {
        "name": (record.get("cv_info") or {}).get("name")
                or record["candidate"]["name_from_email"] or "Unknown",
        "email": record["candidate"]["email_from_email"],
        "message_id": record.get("message_id"),
        "cv_file": record.get("cv_file"),
    }
    outcome = {"candidate": candidate, "job_title": job.get("job_title"),
               "analysed": False, "results": [], "summary": {}}
    if not record.get("cv_info"):
        outcome["note"] = ("This CV could not be analysed automatically "
                           "(unreadable file or AI extraction failed). "
                           "It needs manual review by HR.")
        return outcome

    embedder = embedder or get_embedder()
    profile = {"info": record["cv_info"], "text": record.get("cv_text", ""),
               "lines": cv_lines(record.get("cv_text", ""))}
    for requirement in job["requirements"]:
        matcher = MATCHERS[requirement["category"]]
        if requirement["category"] == "experience":
            result = matcher(requirement, profile, embedder, today or date.today())
        else:
            result = matcher(requirement, profile, embedder)
        outcome["results"].append(result)

    statuses = [r["status"] for r in outcome["results"]]
    outcome["analysed"] = True
    outcome["summary"] = {"meets": statuses.count(MEETS),
                          "partially_meets": statuses.count(PARTIAL),
                          "does_not_meet": statuses.count(NOT_MET)}
    return outcome


def format_match(outcome):
    """Readable report for one candidate."""
    lines = [f"Candidate: {outcome['candidate']['name']} "
             f"<{outcome['candidate']['email']}>  |  Job: {outcome['job_title']}"]
    if not outcome["analysed"]:
        return "\n".join(lines + [outcome["note"]])
    for r in outcome["results"]:
        priority = "MUST-HAVE" if r["must_have"] else "optional"
        lines.append(f"\n{r['requirement_id']} [{priority}] {r['requirement']}")
        lines.append(f"   Result  : {r['status']}")
        lines.append(f"   Why     : {r['explanation']}")
        for quote in r["evidence"]:
            lines.append(f"   Evidence: {quote}")
    s = outcome["summary"]
    lines.append(f"\nSummary: {s['meets']} met, {s['partially_meets']} partially met, "
                 f"{s['does_not_meet']} not met")
    return "\n".join(lines)


if __name__ == "__main__":
    from src.job_description import JobDescriptionError, load_job
    from src.text_extractor import load_cv_records

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        current_job = load_job()
    except JobDescriptionError as e:
        raise SystemExit(f"Job description problem: {e}")
    stored = load_cv_records()
    if not stored:
        print("No stored CVs. Run 'python main.py' first.")
    for cv_record in stored:
        print("\n" + "=" * 78)
        print(format_match(match_candidate(current_job, cv_record)))
