"""Step 6 tests: job description files and the Must-Have review.

Run:  python -m unittest tests.test_job_description -v
"""
import copy
import json
import os
import tempfile
import unittest

from src import job_description as jd

SAMPLE_JOB = {
    "job_title": "Python Developer",
    "requirements": [
        {"id": "R1", "category": "skill", "requirement": "Python", "must_have": True},
        {"id": "R2", "category": "skill", "requirement": "Docker", "must_have": False},
        {"id": "R3", "category": "experience", "requirement": "2 years of Python",
         "must_have": True, "min_years": 2},
        {"id": "R4", "category": "language", "requirement": "English", "must_have": False},
    ],
}


def must_ids(job):
    return [r["id"] for r in jd.split_by_priority(job)[0]]


class SampleFilesTests(unittest.TestCase):
    """The sample files shipped with the project must be valid."""

    def test_job_description_text_loads(self):
        text = jd.load_job_description()
        self.assertIn("Python Developer", text)
        self.assertIn("MUST-HAVE REQUIREMENTS", text)
        self.assertIn("NICE-TO-HAVE REQUIREMENTS", text)

    def test_requirements_cover_all_five_categories(self):
        job = jd.load_job()
        self.assertEqual({r["category"] for r in job["requirements"]}, set(jd.CATEGORIES))

    def test_sample_has_must_have_and_optional_requirements(self):
        must, optional = jd.split_by_priority(jd.load_job())
        self.assertTrue(must)
        self.assertTrue(optional)


class ValidationTests(unittest.TestCase):
    def broken(self, **changes):
        job = copy.deepcopy(SAMPLE_JOB)
        job["requirements"][0].update(changes)
        return job

    def test_valid_job_passes(self):
        self.assertIs(jd.validate_job(SAMPLE_JOB), SAMPLE_JOB)

    def test_missing_requirements_list(self):
        with self.assertRaises(jd.JobDescriptionError):
            jd.validate_job({"job_title": "x"})

    def test_unknown_category(self):
        with self.assertRaisesRegex(jd.JobDescriptionError, "category"):
            jd.validate_job(self.broken(category="hobby"))

    def test_must_have_must_be_true_or_false(self):
        with self.assertRaisesRegex(jd.JobDescriptionError, "must_have"):
            jd.validate_job(self.broken(must_have="yes"))

    def test_empty_requirement_text(self):
        with self.assertRaisesRegex(jd.JobDescriptionError, "non-empty"):
            jd.validate_job(self.broken(requirement="  "))

    def test_duplicate_id(self):
        with self.assertRaisesRegex(jd.JobDescriptionError, "Duplicate"):
            jd.validate_job(self.broken(id="R2"))

    def test_negative_years(self):
        with self.assertRaisesRegex(jd.JobDescriptionError, "min_years"):
            jd.validate_job(self.broken(min_years=-1))


class MustHaveSelectionTests(unittest.TestCase):
    def setUp(self):
        self.job = copy.deepcopy(SAMPLE_JOB)

    def test_split_by_priority(self):
        must, optional = jd.split_by_priority(self.job)
        self.assertEqual([r["id"] for r in must], ["R1", "R3"])
        self.assertEqual([r["id"] for r in optional], ["R2", "R4"])

    def test_toggle_switches_both_ways(self):
        jd.toggle_must_have(self.job, ["R1", "r2"])  # ids are not case sensitive
        self.assertEqual(must_ids(self.job), ["R2", "R3"])

    def test_toggle_with_unknown_id_changes_nothing(self):
        with self.assertRaises(jd.JobDescriptionError):
            jd.toggle_must_have(self.job, ["R2", "R99"])
        self.assertEqual(must_ids(self.job), ["R1", "R3"])

    def test_set_must_haves_replaces_the_selection(self):
        jd.set_must_haves(self.job, ["R4"])
        self.assertEqual(must_ids(self.job), ["R4"])


class FileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "job", "requirements.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_and_load(self):
        jd.save_job(SAMPLE_JOB, self.path)
        self.assertEqual(jd.load_job(self.path), SAMPLE_JOB)

    def test_missing_file(self):
        with self.assertRaisesRegex(jd.JobDescriptionError, "not found"):
            jd.load_job(self.path)

    def test_invalid_json(self):
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "w") as f:
            f.write("{oops")
        with self.assertRaisesRegex(jd.JobDescriptionError, "not valid JSON"):
            jd.load_job(self.path)

    def test_empty_job_description_text(self):
        text_path = os.path.join(self.tmp.name, "jd.txt")
        open(text_path, "w").close()
        with self.assertRaisesRegex(jd.JobDescriptionError, "empty"):
            jd.load_job_description(text_path)


class ReviewScreenTests(unittest.TestCase):
    """Drive the HR review screen with prepared answers instead of a keyboard."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "requirements.json")
        self.job = copy.deepcopy(SAMPLE_JOB)
        self.printed = []

    def tearDown(self):
        self.tmp.cleanup()

    def run_review(self, answers):
        answers = iter(answers)
        return jd.review_must_haves(self.job, self.path,
                                    input_func=lambda prompt: next(answers),
                                    output=self.printed.append)

    def test_hr_changes_selection_and_saves(self):
        self.assertTrue(self.run_review(["R2, R3", "s"]))
        with open(self.path, encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(must_ids(saved), ["R1", "R2"])

    def test_quit_does_not_save(self):
        self.assertFalse(self.run_review(["R2", "q"]))
        self.assertFalse(os.path.exists(self.path))

    def test_wrong_id_is_explained_and_review_continues(self):
        self.run_review(["R99", "s"])
        self.assertTrue(any("Not changed" in str(line) for line in self.printed))
        self.assertEqual(must_ids(jd.load_job(self.path)), ["R1", "R3"])

    def test_warning_when_no_must_have_is_left(self):
        self.run_review(["R1 R3", "s"])
        self.assertTrue(any("no Must-Have" in str(line) for line in self.printed))

    def test_format_marks_must_have_and_optional(self):
        text = jd.format_job(self.job)
        self.assertIn("R1   [MUST-HAVE] Python", text)
        self.assertIn("R2   [optional ] Docker", text)
        self.assertIn("2 Must-Have, 2 optional", text)


if __name__ == "__main__":
    unittest.main()
