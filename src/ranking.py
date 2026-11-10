"""Step 9 - Must-Have filtering and candidate ranking.

Takes the per-requirement results of Step 8 and sorts candidates into
three groups for HR:

    1. Meets all Must-Haves       every Must-Have is "Meets".
                                  Ranked by overall score, best first.
    2. Needs further review       no Must-Have is failed, but at least one
                                  is only partially met; also CVs that could
                                  not be read automatically.
    3. Does not meet Must-Haves   at least one Must-Have is "Does Not Meet".

IMPORTANT: this is a recommendation, not a decision. Nobody is rejected by
the program. Every candidate stays in the report with the reasons and the
evidence, so HR can check and overrule any result.

Score (0-100) = points earned / points possible, where
    Meets = 1 point, Partially Meets = 0.5, Does Not Meet = 0
    and a Must-Have counts MUST_HAVE_WEIGHT times an optional requirement.

Run on the stored CVs:  python -m src.ranking
"""
import json
import logging
import os
from datetime import datetime, timezone

from src.config import MUST_HAVE_WEIGHT, REPORT_PATH
from src.matching import MEETS, NOT_MET, PARTIAL, match_candidate

logger = logging.getLogger(__name__)

QUALIFIED = "Meets all Must-Have requirements"
NEEDS_REVIEW = "Needs further review"
NOT_QUALIFIED = "Does not meet one or more Must-Have requirements"

POINTS = {MEETS: 1.0, PARTIAL: 0.5, NOT_MET: 0.0}
DISCLAIMER = ("These results are recommendations to support HR. They are not "
              "hiring decisions. Please review the evidence before deciding.")


def calculate_score(results):
    """Overall match score from 0 to 100 (Must-Haves weigh more)."""
    earned = possible = 0.0
    for result in results:
        weight = MUST_HAVE_WEIGHT if result["must_have"] else 1
        earned += POINTS[result["status"]] * weight
        possible += weight
    return round(100 * earned / possible, 1) if possible else 0.0


def classify(outcome):
    """Decide the group of one candidate and explain why.

    `outcome` is the result of matching.match_candidate().
    """
    candidate = {
        "candidate": outcome["candidate"],
        "score": None,
        "rank": None,
        "failed_must_haves": [],
        "partial_must_haves": [],
        "summary": outcome["summary"],
        "results": outcome["results"],
    }
    if not outcome["analysed"]:
        candidate["group"] = NEEDS_REVIEW
        candidate["explanation"] = outcome["note"]
        return candidate

    must_haves = [r for r in outcome["results"] if r["must_have"]]
    failed = [r for r in must_haves if r["status"] == NOT_MET]
    partial = [r for r in must_haves if r["status"] == PARTIAL]
    candidate["score"] = calculate_score(outcome["results"])
    candidate["failed_must_haves"] = [r["requirement_id"] for r in failed]
    candidate["partial_must_haves"] = [r["requirement_id"] for r in partial]

    def listed(items):
        return "; ".join(f"{r['requirement_id']} {r['requirement']}" for r in items)

    met = len(must_haves) - len(failed) - len(partial)
    if failed:
        candidate["group"] = NOT_QUALIFIED
        candidate["explanation"] = (
            f"No evidence was found for {len(failed)} of {len(must_haves)} "
            f"Must-Have requirement(s): {listed(failed)}.")
        if partial:
            candidate["explanation"] += f" Partially met: {listed(partial)}."
    elif partial:
        candidate["group"] = NEEDS_REVIEW
        candidate["explanation"] = (
            f"No Must-Have is missing, but {len(partial)} of {len(must_haves)} "
            f"are only partially met and need a human decision: {listed(partial)}.")
    else:
        candidate["group"] = QUALIFIED
        optional = [r for r in outcome["results"] if not r["must_have"]]
        optional_met = sum(r["status"] == MEETS for r in optional)
        candidate["explanation"] = (
            f"All {met} Must-Have requirement(s) are met with evidence from the CV. "
            f"{optional_met} of {len(optional)} optional requirement(s) are fully met.")
    return candidate


def _ranking_key(candidate):
    """Best first: higher score, then more requirements fully met."""
    return (-(candidate["score"] or 0), -candidate["summary"].get("meets", 0),
            candidate["candidate"]["name"].lower())


def rank_candidates(outcomes, job_title=None):
    """Group and rank all candidates. Returns the screening report."""
    groups = {QUALIFIED: [], NEEDS_REVIEW: [], NOT_QUALIFIED: []}
    for outcome in outcomes:
        candidate = classify(outcome)
        groups[candidate["group"]].append(candidate)
    for members in groups.values():
        members.sort(key=_ranking_key)
    # Only candidates who meet every Must-Have receive a rank number.
    for position, candidate in enumerate(groups[QUALIFIED], start=1):
        candidate["rank"] = position
    return {
        "job_title": job_title,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
        "total_candidates": len(outcomes),
        "qualified": groups[QUALIFIED],
        "needs_review": groups[NEEDS_REVIEW],
        "not_qualified": groups[NOT_QUALIFIED],
    }


def screen_candidates(job, records, embedder=None, today=None):
    """Steps 8 and 9 together: match every CV record, then group and rank."""
    outcomes = [match_candidate(job, record, embedder, today) for record in records]
    return rank_candidates(outcomes, job.get("job_title"))


def save_report(report, path=REPORT_PATH):
    """Store the full report (with all evidence) as JSON for HR."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path


def format_report(report, details=True):
    """Readable screening report."""
    lines = ["=" * 78,
             f"SCREENING REPORT - {report['job_title']}",
             f"{report['total_candidates']} candidate(s) | {report['disclaimer']}",
             "=" * 78]

    def add_group(title, members, note):
        lines.append(f"\n{title} ({len(members)})")
        lines.append(f"  {note}")
        if not members:
            lines.append("  (none)")
        for c in members:
            label = f"#{c['rank']} " if c["rank"] else "- "
            score = "not scored" if c["score"] is None else f"score {c['score']:g}/100"
            lines.append(f"\n  {label}{c['candidate']['name']} "
                         f"<{c['candidate']['email']}> | {score}")
            lines.append(f"     {c['explanation']}")
            if not details:
                continue
            for r in c["results"]:
                if r["status"] == MEETS:
                    continue  # show only what HR needs to look at
                priority = "MUST-HAVE" if r["must_have"] else "optional"
                lines.append(f"     * {r['requirement_id']} [{priority}] {r['status']}: "
                             f"{r['explanation']}")
                lines.extend(f"         Evidence: {quote}" for quote in r["evidence"])

    add_group("1. MEETS ALL MUST-HAVE REQUIREMENTS", report["qualified"],
              "Ranked by overall score, best first.")
    add_group("2. NEEDS FURTHER REVIEW", report["needs_review"],
              "No Must-Have is missing, but something needs a human decision.")
    add_group("3. DOES NOT MEET ONE OR MORE MUST-HAVE REQUIREMENTS", report["not_qualified"],
              "Kept for HR review. Not rejected automatically.")
    return "\n".join(lines)


if __name__ == "__main__":
    from src.job_description import JobDescriptionError, load_job
    from src.text_extractor import load_cv_records

    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
    try:
        current_job = load_job()
    except JobDescriptionError as e:
        raise SystemExit(f"Job description problem: {e}")
    stored = load_cv_records()
    if not stored:
        raise SystemExit("No stored CVs. Run 'python main.py' first.")
    final_report = screen_candidates(current_job, stored)
    print(format_report(final_report))
    print(f"\nFull report with all evidence saved to: {save_report(final_report)}")
