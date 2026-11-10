"""Step 8 tests: embeddings, cosine similarity and CV-to-job matching.

These tests load the real embedding model (all-MiniLM-L6-v2), so the first
run downloads it (about 90 MB) and loading takes up to a minute.
They do not need Ollama or Gmail.

Run:  python -m unittest tests.test_matching -v
"""
import copy
import unittest

from src import matching as m
from src.job_description import load_job
from tests import sample_candidates as sc


def by_id(outcome):
    return {r["requirement_id"]: r for r in outcome["results"]}


def requirement(category, text, must_have=True, **extra):
    return dict({"id": "T1", "category": category, "requirement": text,
                 "must_have": must_have}, **extra)


class MatchingTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = m.get_embedder()   # shared, so the model loads once
        cls.job = load_job()

    def match(self, record, job=None):
        return m.match_candidate(job or self.job, record, self.embedder, sc.TODAY)

    def match_one(self, record, req):
        job = {"job_title": "Test", "requirements": [req]}
        return self.match(record, job)["results"][0]


class EmbeddingTests(MatchingTestCase):
    def test_embedding_is_a_vector_of_384_numbers(self):
        vectors = self.embedder.encode(["Python developer", "Graphic designer"])
        self.assertEqual(len(vectors), 2)
        self.assertEqual(vectors[0].shape, (384,))

    def test_same_text_gives_the_same_vector(self):
        first = self.embedder.encode(["REST API development"])[0]
        second = self.embedder.encode(["REST API development"])[0]
        self.assertTrue((first == second).all())


class CosineSimilarityTests(MatchingTestCase):
    def test_known_values(self):
        self.assertAlmostEqual(m.cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(m.cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(m.cosine_similarity([1, 0], [-1, 0]), -1.0)
        self.assertAlmostEqual(m.cosine_similarity([1, 1], [1, 0]), 0.7071, places=4)

    def test_length_of_the_vector_does_not_matter(self):
        self.assertAlmostEqual(m.cosine_similarity([1, 2, 3], [10, 20, 30]), 1.0)

    def test_zero_vector_gives_zero_instead_of_an_error(self):
        self.assertEqual(m.cosine_similarity([0, 0], [1, 2]), 0.0)

    def test_similar_meaning_scores_higher_than_unrelated(self):
        git, github, photoshop = self.embedder.encode(
            ["Git for version control", "GitHub", "Adobe Photoshop"])
        self.assertGreater(m.cosine_similarity(git, github), 0.55)
        self.assertLess(m.cosine_similarity(git, photoshop), 0.35)

    def test_best_match(self):
        score, text = m.best_match(self.embedder, "Relational databases",
                                   ["Typography", "PostgreSQL", "Branding"])
        self.assertEqual(text, "PostgreSQL")
        self.assertEqual(m.best_match(self.embedder, "anything", []), (0.0, None))


class StrongCandidateTests(MatchingTestCase):
    """A candidate who meets the requirements."""

    def setUp(self):
        self.outcome = self.match(sc.strong())

    def test_meets_every_requirement(self):
        self.assertEqual(self.outcome["summary"],
                         {"meets": 12, "partially_meets": 0, "does_not_meet": 0})

    def test_every_result_has_evidence_copied_from_the_cv(self):
        cv_text = sc.strong()["cv_text"]
        for result in self.outcome["results"]:
            self.assertTrue(result["explanation"])
            for quote in result["evidence"]:
                self.assertIn(quote, cv_text, result["requirement_id"])

    def test_experience_uses_years_calculated_from_dates(self):
        r8 = by_id(self.outcome)["R8"]
        self.assertEqual(r8["status"], m.MEETS)
        self.assertIn("6.2 year(s)", r8["explanation"])
        self.assertTrue(any("Cedar Tech" in quote for quote in r8["evidence"]))


class PartialCandidateTests(MatchingTestCase):
    """A related candidate who uses different wording and falls a bit short."""

    def setUp(self):
        self.results = by_id(self.match(sc.partial()))

    def test_different_wording_is_matched_by_meaning(self):
        # "Python scripting" for "Strong Python programming skills",
        # "GitHub" for "Git for version control": no shared keyword rule.
        for req_id in ("R1", "R4"):
            self.assertEqual(self.results[req_id]["status"], m.MEETS)
            self.assertEqual(self.results[req_id]["method"], "semantic")
            self.assertGreaterEqual(self.results[req_id]["similarity"], 0.55)

    def test_loosely_related_skill_is_partial(self):
        r3 = self.results["R3"]  # has MySQL, requirement names SQL / PostgreSQL
        self.assertEqual(r3["status"], m.PARTIAL)
        self.assertIn("MySQL", r3["explanation"])

    def test_too_few_years_is_partial(self):
        r8 = self.results["R8"]
        self.assertEqual(r8["status"], m.PARTIAL)
        self.assertIn("1.9 year(s)", r8["explanation"])
        self.assertIn("below the minimum of 2", r8["explanation"])

    def test_lower_language_level_is_partial(self):
        self.assertEqual(self.results["R10"]["status"], m.PARTIAL)
        self.assertIn("intermediate", self.results["R10"]["explanation"])

    def test_missing_items_say_no_evidence_found(self):
        for req_id in ("R6", "R12"):  # Docker, certification
            self.assertEqual(self.results[req_id]["status"], m.NOT_MET)
            self.assertEqual(self.results[req_id]["evidence"], [m.NO_EVIDENCE])


class WeakCandidateTests(MatchingTestCase):
    """A candidate from an unrelated field."""

    def setUp(self):
        self.outcome = self.match(sc.weak())
        self.results = by_id(self.outcome)

    def test_only_the_arabic_requirement_is_met(self):
        self.assertEqual(self.outcome["summary"],
                         {"meets": 1, "partially_meets": 0, "does_not_meet": 11})
        self.assertEqual(self.results["R11"]["status"], m.MEETS)

    def test_wish_to_learn_python_is_not_counted_as_a_skill(self):
        # The CV says "Interested in learning Python in the future."
        r1 = self.results["R1"]
        self.assertEqual(r1["status"], m.NOT_MET)
        self.assertEqual(r1["evidence"], [m.NO_EVIDENCE])
        self.assertIn(m.NO_EVIDENCE, r1["explanation"])

    def test_unrelated_experience_does_not_count_even_with_many_years(self):
        r8 = self.results["R8"]  # 7 years, but as a graphic designer
        self.assertEqual(r8["status"], m.NOT_MET)
        self.assertIn("none is related", r8["explanation"])

    def test_lower_degree_in_another_field_does_not_meet(self):
        self.assertEqual(self.results["R9"]["status"], m.NOT_MET)


class RuleTests(MatchingTestCase):
    def test_skill_described_as_basic_is_partial(self):
        record = sc.strong()
        record["cv_text"] = record["cv_text"].replace("Docker,", "Docker (basic),")
        result = self.match_one(record, requirement("skill", "Docker and containerised deployments"))
        self.assertEqual(result["status"], m.PARTIAL)
        self.assertIn("basic", result["explanation"])

    def test_keyword_inside_another_word_does_not_match(self):
        # "Java" must not be found inside "JavaScript".
        self.assertFalse(m._contains_term("Experience with JavaScript", "Java"))
        self.assertTrue(m._contains_term("Experience with Java and SQL", "Java"))
        self.assertTrue(m._contains_term("C++ or C# development", "C++"))
        self.assertTrue(m._contains_term("Building REST APIs", "REST API"))

    def test_experience_without_dates_is_partial(self):
        record = sc.strong()
        record["cv_info"]["stated_years_experience"] = None
        for job in record["cv_info"]["work_experience"]:
            job["start_date"] = job["end_date"] = None
        result = self.match_one(record, self.job["requirements"][7])
        self.assertEqual(result["status"], m.PARTIAL)
        self.assertIn("no dates", result["explanation"])

    def test_no_work_experience_at_all(self):
        record = sc.strong()
        record["cv_info"]["work_experience"] = []
        result = self.match_one(record, self.job["requirements"][7])
        self.assertEqual(result["status"], m.NOT_MET)

    def test_degree_levels(self):
        self.assertEqual(m.degree_level("Bachelor of Science")[1], "Bachelor's")
        self.assertEqual(m.degree_level("B.Sc. Computer Science")[1], "Bachelor's")
        self.assertEqual(m.degree_level("MSc in Data Science")[1], "Master's")
        self.assertEqual(m.degree_level("PhD")[1], "PhD")
        self.assertEqual(m.degree_level("Diploma")[1], "Diploma")
        self.assertEqual(m.degree_level("High school"), (0, None))

    def test_higher_degree_also_meets_a_bachelor_requirement(self):
        record = sc.strong()
        record["cv_info"]["education"][0]["degree"] = "Master of Science"
        record["cv_text"] = record["cv_text"].replace("Bachelor of Science", "Master of Science")
        self.assertEqual(self.match_one(record, self.job["requirements"][8])["status"], m.MEETS)

    def test_right_level_but_unrelated_field_is_partial(self):
        record = sc.strong()
        record["cv_info"]["education"][0]["field"] = "History"
        record["cv_text"] = record["cv_text"].replace("in Computer Science", "in History")
        result = self.match_one(record, self.job["requirements"][8])
        self.assertEqual(result["status"], m.PARTIAL)
        self.assertIn("HR should judge", result["explanation"])

    def test_language_without_required_level_only_needs_to_be_listed(self):
        result = self.match_one(sc.partial(), requirement("language", "English"))
        self.assertEqual(result["status"], m.MEETS)

    def test_claim_without_a_supporting_cv_line_is_never_accepted(self):
        # cv_info says "Kubernetes", but the CV text never mentions it.
        record = sc.strong()
        record["cv_info"]["skills"].append("Kubernetes")
        result = self.match_one(record, requirement("skill", "Kubernetes"))
        self.assertEqual(result["status"], m.NOT_MET)
        self.assertEqual(result["evidence"], [m.NO_EVIDENCE])


class CandidateLevelTests(MatchingTestCase):
    def test_unreadable_cv_is_sent_to_manual_review_not_judged(self):
        outcome = self.match(sc.unreadable())
        self.assertFalse(outcome["analysed"])
        self.assertEqual(outcome["results"], [])
        self.assertIn("manual review", outcome["note"])

    def test_every_result_has_all_fields(self):
        for record in (sc.strong(), sc.partial(), sc.weak()):
            for result in self.match(record)["results"]:
                self.assertIn(result["status"], (m.MEETS, m.PARTIAL, m.NOT_MET))
                self.assertTrue(result["evidence"])
                self.assertTrue(result["explanation"])
                self.assertIn(result["method"], ("rule", "semantic", "rule+semantic"))

    def test_report_text_contains_result_reason_and_evidence(self):
        text = m.format_match(self.match(sc.partial()))
        self.assertIn("Omar Khalil", text)
        self.assertIn("Partially Meets the Requirement", text)
        self.assertIn("Evidence: No evidence found", text)
        self.assertIn("Summary: 4 met, 6 partially met, 2 not met", text)

    def test_matching_does_not_change_the_record(self):
        record = sc.partial()
        before = copy.deepcopy(record)
        self.match(record)
        self.assertEqual(record, before)


if __name__ == "__main__":
    unittest.main()
