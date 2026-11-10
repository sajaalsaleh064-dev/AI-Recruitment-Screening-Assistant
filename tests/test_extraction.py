"""Step 7 tests: AI extraction, JSON validation and the grounding check.

These tests replace the real model with a FAKE one that returns prepared
answers, so they run in a second and do not need Ollama. They check what
our code does with good, incomplete and wrong answers from a model.

Run:  python -m unittest tests.test_extraction -v

To test the real model, use:  python -m tests.live_extraction_check
"""
import os
import unittest
from datetime import date

from src import ai_extraction as ai
from src import ollama_client as oc
from src.job_description import validate_job

SAMPLES = os.path.join(os.path.dirname(__file__), "sample_data")
TODAY = date(2026, 11, 1)


def sample(name):
    with open(os.path.join(SAMPLES, name), encoding="utf-8") as f:
        return f.read()


def fake_llm(answer):
    """Build a fake model that always returns `answer`."""
    def llm(system_prompt, user_prompt, schema):
        llm.calls.append(user_prompt)
        return answer
    llm.calls = []
    return llm


GOOD_CV_ANSWER = {
    "name": "Layla Haddad",
    "skills": ["Python", "SQL", "FastAPI", "Flask", "REST APIs", "PostgreSQL",
               "Git", "Docker", "AWS", "pytest", "python"],  # duplicate on purpose
    "work_experience": [
        {"job_title": "Python Developer", "company": "Cedar Tech",
         "start_date": "Mar 2022", "end_date": "Present",
         "description": "Built and maintained REST APIs with FastAPI."},
        {"job_title": "Junior Software Developer", "company": "Petra Systems",
         "start_date": "Sep 2020", "end_date": "Feb 2022",
         "description": "Developed Flask web services."},
    ],
    "stated_years_experience": 4,
    "education": [{"degree": "Bachelor of Science", "field": "Computer Science",
                   "institution": "University of Jordan", "year": "2020"}],
    "languages": [{"language": "Arabic", "level": "native"},
                  {"language": "English", "level": "fluent"}],
    "certifications": ["AWS Certified Cloud Practitioner"],
}


class JsonParsingTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(oc.parse_json_object('{"a": 1}'), {"a": 1})

    def test_json_inside_code_fence_and_text(self):
        answer = 'Here is the result:\n```json\n{"skills": ["Python"]}\n```\nDone.'
        self.assertEqual(oc.parse_json_object(answer), {"skills": ["Python"]})

    def test_invalid_answers_raise_a_clear_error(self):
        for bad in ["", "I cannot help with that", '{"a": 1', "[1, 2]", None]:
            with self.assertRaises(oc.OllamaError):
                oc.parse_json_object(bad)

    def test_retry_after_invalid_json(self):
        answers = iter(["not json at all", '{"ok": true}'])
        result = oc.generate_json("s", "u", chat=lambda *args: next(answers))
        self.assertEqual(result, {"ok": True})

    def test_gives_up_after_retries(self):
        with self.assertRaisesRegex(oc.OllamaError, "No valid JSON after 2"):
            oc.generate_json("s", "u", chat=lambda *args: "still not json")


class CvExtractionTests(unittest.TestCase):
    def setUp(self):
        self.text = sample("cv_strong.txt")
        self.info = ai.extract_cv_information(self.text, fake_llm(GOOD_CV_ANSWER), TODAY)

    def test_cv_text_is_sent_to_the_model(self):
        llm = fake_llm(GOOD_CV_ANSWER)
        ai.extract_cv_information(self.text, llm, TODAY)
        self.assertIn("Layla Haddad", llm.calls[0])

    def test_all_sections_are_returned(self):
        self.assertEqual(self.info["name"], "Layla Haddad")
        self.assertIn("FastAPI", self.info["skills"])
        self.assertEqual(len(self.info["work_experience"]), 2)
        self.assertEqual(self.info["education"][0]["institution"], "University of Jordan")
        self.assertEqual(self.info["languages"][1], {"language": "English", "level": "fluent"})
        self.assertEqual(self.info["certifications"], ["AWS Certified Cloud Practitioner"])
        self.assertEqual(self.info["unverified"], [])

    def test_duplicate_skills_are_removed(self):
        self.assertEqual(len([s for s in self.info["skills"] if s.lower() == "python"]), 1)

    def test_years_are_calculated_from_dates(self):
        # Sep 2020 -> Nov 2026 with no gap = 75 months = 6.2 years
        self.assertEqual(self.info["calculated_years_experience"], 6.2)
        self.assertEqual(self.info["stated_years_experience"], 4)

    def test_empty_cv_text_is_rejected(self):
        with self.assertRaises(ValueError):
            ai.extract_cv_information("   ", fake_llm(GOOD_CV_ANSWER))


class NoInventedInformationTests(unittest.TestCase):
    """The grounding check: nothing may be added that the CV does not say."""

    def setUp(self):
        self.text = sample("cv_weak.txt")  # a graphic designer's CV

    def extract(self, **answer):
        base = {"name": "Rana Mansour", "skills": ["Figma"], "work_experience": [],
                "stated_years_experience": None, "education": [], "languages": [],
                "certifications": []}
        base.update(answer)
        return ai.extract_cv_information(self.text, fake_llm(base), TODAY)

    def test_invented_skill_is_moved_to_unverified(self):
        info = self.extract(skills=["Figma", "Kubernetes"])
        self.assertEqual(info["skills"], ["Figma"])
        self.assertIn({"category": "skill", "value": "Kubernetes"}, info["unverified"])

    def test_invented_certification_degree_job_and_language(self):
        info = self.extract(
            certifications=["PMP"],
            education=[{"degree": "Master of Science", "field": "Data Science",
                        "institution": "MIT", "year": "2022"}],
            work_experience=[{"job_title": "Senior Backend Engineer", "company": "Google",
                              "start_date": "2020", "end_date": "2023", "description": None}],
            languages=[{"language": "German", "level": "fluent"}],
        )
        self.assertEqual(info["certifications"], [])
        self.assertEqual(info["education"], [])
        self.assertEqual(info["work_experience"], [])
        self.assertEqual(info["languages"], [])
        self.assertEqual({u["category"] for u in info["unverified"]},
                         {"certification", "education", "work_experience", "language"})

    def test_invented_years_are_removed(self):
        self.assertIsNone(self.extract(stated_years_experience=12)["stated_years_experience"])
        self.assertEqual(self.extract(stated_years_experience=5)["stated_years_experience"], 5)

    def test_stated_years_are_found_even_if_the_model_misses_them(self):
        self.assertEqual(self.extract(stated_years_experience=None)["stated_years_experience"], 5)
        self.assertEqual(ai.find_stated_years("Developer with 2+ years of hands-on experience"), 2)
        self.assertEqual(ai.find_stated_years("3 yrs experience in sales"), 3)
        self.assertIsNone(ai.find_stated_years("Worked for many years. Experience: Python"))

    def test_wrong_name_is_removed(self):
        self.assertIsNone(self.extract(name="John Smith")["name"])


class MissingAndInconsistentAnswerTests(unittest.TestCase):
    def test_empty_answer_gives_empty_but_complete_structure(self):
        info = ai.extract_cv_information("Some CV text", fake_llm({}), TODAY)
        self.assertEqual(info, {
            "name": None, "skills": [], "work_experience": [],
            "stated_years_experience": None, "calculated_years_experience": None,
            "education": [], "languages": [], "certifications": [], "unverified": [],
        })

    def test_wrong_types_are_repaired(self):
        text = "Sara. Skills: Python. Languages: English. 3 years of experience."
        info = ai.extract_cv_information(text, fake_llm({
            "name": "N/A",
            "skills": "Python",                       # string instead of list
            "languages": ["English"],                 # strings instead of objects
            "stated_years_experience": "3 years",     # text instead of number
            "certifications": None,
            "work_experience": "none",
            "education": [None, "x"],
        }), TODAY)
        self.assertIsNone(info["name"])
        self.assertEqual(info["skills"], ["Python"])
        self.assertEqual(info["languages"], [{"language": "English", "level": None}])
        self.assertEqual(info["stated_years_experience"], 3)
        self.assertEqual(info["certifications"], [])
        self.assertEqual(info["work_experience"], [])
        self.assertEqual(info["education"], [])


class YearsCalculationTests(unittest.TestCase):
    def test_date_formats(self):
        self.assertEqual(ai.parse_month("Jun 2024"), (2024, 6))
        self.assertEqual(ai.parse_month("September 2020"), (2020, 9))
        self.assertEqual(ai.parse_month("03/2021"), (2021, 3))
        self.assertEqual(ai.parse_month("2019"), (2019, 1))
        self.assertEqual(ai.parse_month("2019", is_end=True), (2019, 12))
        self.assertEqual(ai.parse_month("Present", TODAY), (2026, 11))
        self.assertIsNone(ai.parse_month("unknown"))
        self.assertIsNone(ai.parse_month(None))

    def test_overlapping_jobs_are_not_counted_twice(self):
        jobs = [{"start_date": "Jan 2022", "end_date": "Dec 2022"},
                {"start_date": "Jul 2022", "end_date": "Jun 2023"}]
        self.assertEqual(ai.calculate_years_experience(jobs, TODAY), 1.5)

    def test_missing_or_impossible_dates_are_ignored(self):
        jobs = [{"start_date": None, "end_date": "2023"},
                {"start_date": "Jan 2025", "end_date": "Jan 2024"},   # ends before start
                {"start_date": "Jan 2030", "end_date": "Dec 2031"}]   # in the future
        self.assertIsNone(ai.calculate_years_experience(jobs, TODAY))


class JobExtractionTests(unittest.TestCase):
    ANSWER = {
        "job_title": "Data Analyst",
        "required_skills": [
            {"requirement": "Advanced Excel and SQL", "must_have": True, "min_years": None},
            {"requirement": "Python with pandas", "must_have": False, "min_years": None},
            {"requirement": "Machine learning with TensorFlow", "must_have": True,
             "min_years": None},  # NOT in the job text
        ],
        "required_experience": [
            {"requirement": "Minimum 3 years of experience in data analysis",
             "must_have": True, "min_years": 3}],
        "education": [{"requirement": "Bachelor's degree in Statistics, Mathematics or "
                                      "Computer Science", "must_have": True}],
        "languages": [{"requirement": "Fluent English", "must_have": "yes"},  # wrong type
                      "French"],                                              # plain string
        "certifications": None,
    }

    def setUp(self):
        self.text = sample("job_data_analyst.txt")
        self.info = ai.extract_job_requirements(self.text, fake_llm(self.ANSWER))

    def test_requirements_are_grouped_by_category(self):
        self.assertEqual(self.info["job_title"], "Data Analyst")
        self.assertEqual([r["requirement"] for r in self.info["required_skills"]],
                         ["Advanced Excel and SQL", "Python with pandas"])
        self.assertEqual(self.info["required_experience"][0]["min_years"], 3)
        self.assertEqual(self.info["certifications"], [])

    def test_must_have_and_optional_are_kept(self):
        flags = {r["requirement"]: r["must_have"] for r in self.info["required_skills"]}
        self.assertTrue(flags["Advanced Excel and SQL"])
        self.assertFalse(flags["Python with pandas"])

    def test_unclear_must_have_defaults_to_true_for_hr_review(self):
        self.assertTrue(all(r["must_have"] for r in self.info["languages"]))

    def test_invented_requirement_is_not_used(self):
        self.assertIn({"category": "skill", "value": "Machine learning with TensorFlow"},
                      self.info["unverified"])

    def test_requirement_repeated_under_skills_is_kept_once(self):
        answer = dict(self.ANSWER, required_skills=self.ANSWER["required_skills"] + [
            {"requirement": "French", "must_have": False, "min_years": None}])
        info = ai.extract_job_requirements(self.text, fake_llm(answer))
        self.assertNotIn("French", [r["requirement"] for r in info["required_skills"]])
        self.assertIn("French", [r["requirement"] for r in info["languages"]])

    def test_conversion_to_reviewable_job_format(self):
        job = ai.job_from_extraction(self.info)
        validate_job(job)  # same format as job/job_requirements.json
        self.assertEqual([r["id"] for r in job["requirements"]],
                         ["R1", "R2", "R3", "R4", "R5", "R6"])
        self.assertEqual(job["requirements"][2]["category"], "experience")
        self.assertEqual(job["requirements"][2]["min_years"], 3)


class GroundingHelperTests(unittest.TestCase):
    def test_exact_and_word_by_word_matches(self):
        text = "Skills: Python, REST APIs, unit testing.\nC++ and C# basics"
        for phrase in ["python", "REST API", "Unit  Testing", "C++", "C#",
                       "testing unit"]:
            self.assertTrue(ai.appears_in_text(phrase, text), phrase)

    def test_absent_phrases(self):
        text = "Skills: Python, REST APIs"
        for phrase in ["Java Spring", "Kubernetes", "", None]:
            self.assertFalse(ai.appears_in_text(phrase, text), phrase)


if __name__ == "__main__":
    unittest.main()
