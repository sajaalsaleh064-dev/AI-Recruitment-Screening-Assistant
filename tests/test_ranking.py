"""Step 9 tests: Must-Have filtering, grouping and ranking.

Uses the three sample candidates and the real embedding model
(no Ollama or Gmail needed).

Run:  python -m unittest tests.test_ranking -v
"""
import copy
import json
import os
import tempfile
import unittest

from src import ranking as rk
from src.job_description import load_job, toggle_must_have
from src.matching import MEETS, NOT_MET, PARTIAL, get_embedder
from tests import sample_candidates as sc


def result(req_id, status, must_have):
    return {"requirement_id": req_id, "category": "skill", "requirement": f"Req {req_id}",
            "must_have": must_have, "status": status, "evidence": ["line"],
            "similarity": None, "method": "rule", "explanation": "because"}


def outcome(name, results):
    statuses = [r["status"] for r in results]
    return {"candidate": {"name": name, "email": f"{name.lower()}@example.com"},
            "job_title": "Test job", "analysed": True, "results": results,
            "summary": {"meets": statuses.count(MEETS),
                        "partially_meets": statuses.count(PARTIAL),
                        "does_not_meet": statuses.count(NOT_MET)}}


def names(group):
    return [c["candidate"]["name"] for c in group]


class ScoreTests(unittest.TestCase):
    def test_all_met_is_100_and_none_met_is_0(self):
        self.assertEqual(rk.calculate_score([result("R1", MEETS, True)]), 100)
        self.assertEqual(rk.calculate_score([result("R1", NOT_MET, True)]), 0)
        self.assertEqual(rk.calculate_score([]), 0)

    def test_partial_counts_half(self):
        self.assertEqual(rk.calculate_score([result("R1", PARTIAL, False)]), 50)

    def test_must_have_weighs_twice_an_optional(self):
        # Must-Have met (2 of 2 points) + optional not met (0 of 1) = 2/3
        missing_optional = [result("R1", MEETS, True), result("R2", NOT_MET, False)]
        # Must-Have not met (0 of 2) + optional met (1 of 1) = 1/3
        missing_must = [result("R1", NOT_MET, True), result("R2", MEETS, False)]
        self.assertEqual(rk.calculate_score(missing_optional), 66.7)
        self.assertEqual(rk.calculate_score(missing_must), 33.3)


class ClassifyTests(unittest.TestCase):
    def test_all_must_haves_met_is_qualified_even_without_optionals(self):
        c = rk.classify(outcome("A", [result("R1", MEETS, True), result("R2", NOT_MET, False)]))
        self.assertEqual(c["group"], rk.QUALIFIED)
        self.assertIn("All 1 Must-Have", c["explanation"])

    def test_partial_must_have_needs_review(self):
        c = rk.classify(outcome("A", [result("R1", MEETS, True), result("R2", PARTIAL, True)]))
        self.assertEqual(c["group"], rk.NEEDS_REVIEW)
        self.assertEqual(c["partial_must_haves"], ["R2"])
        self.assertIn("R2 Req R2", c["explanation"])

    def test_failed_must_have_is_separated_with_the_reason(self):
        c = rk.classify(outcome("A", [result("R1", NOT_MET, True), result("R2", PARTIAL, True),
                                      result("R3", MEETS, False)]))
        self.assertEqual(c["group"], rk.NOT_QUALIFIED)
        self.assertEqual(c["failed_must_haves"], ["R1"])
        self.assertIn("R1 Req R1", c["explanation"])
        self.assertIn("Partially met: R2", c["explanation"])

    def test_partial_optional_does_not_block_qualification(self):
        c = rk.classify(outcome("A", [result("R1", MEETS, True), result("R2", PARTIAL, False)]))
        self.assertEqual(c["group"], rk.QUALIFIED)

    def test_unanalysed_cv_goes_to_review_without_a_score(self):
        c = rk.classify({"candidate": {"name": "Scan", "email": "s@example.com"},
                         "analysed": False, "results": [], "summary": {},
                         "note": "needs manual review by HR"})
        self.assertEqual(c["group"], rk.NEEDS_REVIEW)
        self.assertIsNone(c["score"])


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.report = rk.rank_candidates([
            outcome("Good", [result("R1", MEETS, True), result("R2", NOT_MET, False)]),
            outcome("Best", [result("R1", MEETS, True), result("R2", MEETS, False)]),
            outcome("Better", [result("R1", MEETS, True), result("R2", PARTIAL, False)]),
            outcome("Unsure", [result("R1", PARTIAL, True), result("R2", MEETS, False)]),
            outcome("Missing", [result("R1", NOT_MET, True), result("R2", MEETS, False)]),
        ], "Test job")

    def test_qualified_candidates_are_ranked_best_first(self):
        self.assertEqual(names(self.report["qualified"]), ["Best", "Better", "Good"])
        self.assertEqual([c["rank"] for c in self.report["qualified"]], [1, 2, 3])

    def test_other_groups_are_kept_separate_and_unranked(self):
        self.assertEqual(names(self.report["needs_review"]), ["Unsure"])
        self.assertEqual(names(self.report["not_qualified"]), ["Missing"])
        self.assertIsNone(self.report["needs_review"][0]["rank"])
        self.assertIsNone(self.report["not_qualified"][0]["rank"])

    def test_nobody_disappears_from_the_report(self):
        total = sum(len(self.report[g]) for g in ("qualified", "needs_review", "not_qualified"))
        self.assertEqual(total, self.report["total_candidates"])
        self.assertEqual(total, 5)

    def test_report_states_that_it_is_not_a_decision(self):
        self.assertIn("not hiring decisions", self.report["disclaimer"])
        self.assertIn("Not rejected automatically", rk.format_report(self.report))

    def test_report_can_be_saved_as_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = rk.save_report(self.report, os.path.join(folder, "out", "report.json"))
            with open(path, encoding="utf-8") as f:
                self.assertEqual(json.load(f)["qualified"][0]["candidate"]["name"], "Best")


class SampleCandidatesTests(unittest.TestCase):
    """The whole of Steps 8 and 9 on the three sample CVs."""

    @classmethod
    def setUpClass(cls):
        cls.embedder = get_embedder()
        cls.job = load_job()

    def screen(self, records, job=None):
        return rk.screen_candidates(job or self.job, records, self.embedder, sc.TODAY)

    def test_each_sample_lands_in_the_expected_group(self):
        report = self.screen([sc.weak(), sc.partial(), sc.strong(), sc.unreadable()])
        self.assertEqual(names(report["qualified"]), ["Layla Haddad"])
        self.assertEqual(names(report["needs_review"]), ["Omar Khalil", "Scan Only"])
        self.assertEqual(names(report["not_qualified"]), ["Rana Mansour"])

    def test_explanations_name_the_requirements_involved(self):
        report = self.screen([sc.weak(), sc.partial(), sc.strong()])
        self.assertEqual(report["qualified"][0]["score"], 100)
        self.assertEqual(report["needs_review"][0]["partial_must_haves"],
                         ["R2", "R3", "R8", "R10"])
        self.assertEqual(report["not_qualified"][0]["failed_must_haves"],
                         ["R1", "R2", "R3", "R4", "R8", "R9", "R10"])

    def test_two_qualified_candidates_are_ordered_by_score(self):
        second = sc.strong()  # same profile without the optional certificate
        second["cv_info"]["name"] = "Second Candidate"
        second["cv_info"]["certifications"] = []
        report = self.screen([second, sc.strong()])
        self.assertEqual(names(report["qualified"]), ["Layla Haddad", "Second Candidate"])
        self.assertGreater(report["qualified"][0]["score"], report["qualified"][1]["score"])

    def test_hr_changing_a_must_have_changes_the_groups(self):
        job = copy.deepcopy(self.job)
        toggle_must_have(job, ["R6"])  # HR makes Docker a Must-Have
        report = self.screen([sc.partial(), sc.strong()], job)
        self.assertEqual(names(report["qualified"]), ["Layla Haddad"])
        self.assertEqual(names(report["not_qualified"]), ["Omar Khalil"])
        self.assertIn("R6", report["not_qualified"][0]["failed_must_haves"])


if __name__ == "__main__":
    unittest.main()
