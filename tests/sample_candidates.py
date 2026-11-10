"""Three fictional candidates used by the matching and ranking tests.

Each one is a CV record in the same shape the program stores after Step 7:
the CV text (tests/sample_data/*.txt) plus the structured "cv_info".
The cv_info below is what the local LLM returned for these CVs, saved here
so the tests do not need Ollama.

    strong  -> meets the Python Developer job
    partial -> related profile, different wording, slightly too little experience
    weak    -> unrelated profile (graphic designer)
"""
import os
from datetime import date

SAMPLES = os.path.join(os.path.dirname(__file__), "sample_data")
TODAY = date(2026, 11, 1)  # fixed date so "years of experience" never changes


def _read(name):
    with open(os.path.join(SAMPLES, name), encoding="utf-8") as f:
        return f.read()


def _record(file_name, sender_name, email, cv_info):
    cv_info = dict({"stated_years_experience": None, "certifications": [],
                    "unverified": []}, **cv_info)
    return {
        "message_id": f"sample-{file_name}",
        "candidate": {"name_from_email": sender_name, "email_from_email": email,
                      "emails_in_cv": [], "phones_in_cv": []},
        "cv_file": file_name,
        "original_filename": file_name,
        "extraction": {"status": "ok"},
        "cv_text": _read(file_name),
        "cv_info": cv_info,
    }


def strong():
    return _record("cv_strong.txt", "Layla Haddad", "layla.haddad.test@example.com", {
        "name": "Layla Haddad",
        "skills": ["Python", "SQL", "Bash", "FastAPI", "Flask", "REST APIs", "PostgreSQL",
                   "Redis", "Git", "GitHub", "Docker", "AWS", "EC2", "S3", "Linux",
                   "pytest", "unit testing"],
        "work_experience": [
            {"job_title": "Python Developer", "company": "Cedar Tech",
             "start_date": "Mar 2022", "end_date": "Present",
             "description": "Built and maintained REST APIs with FastAPI serving 20,000 daily users."},
            {"job_title": "Junior Software Developer", "company": "Petra Systems",
             "start_date": "Sep 2020", "end_date": "Feb 2022",
             "description": "Developed Flask web services and wrote pytest test suites."},
        ],
        "stated_years_experience": 4,
        "education": [{"degree": "Bachelor of Science", "field": "Computer Science",
                       "institution": "University of Jordan", "year": "2020"}],
        "languages": [{"language": "Arabic", "level": "native"},
                      {"language": "English", "level": "fluent"}],
        "certifications": ["AWS Certified Cloud Practitioner"],
    })


def partial():
    return _record("cv_partial.txt", "Omar Khalil", "omar.khalil.test@example.com", {
        "name": "Omar Khalil",
        "skills": ["Python scripting", "Django", "building web services", "MySQL",
                   "GitHub", "Linux", "HTML", "CSS"],
        "work_experience": [
            {"job_title": "Software Engineer", "company": "Jordan Valley Apps",
             "start_date": "Jan 2025", "end_date": "Present",
             "description": "Develop server-side features for a Django e-commerce site."},
        ],
        "education": [{"degree": "Bachelor's degree", "field": "Software Engineering",
                       "institution": "Yarmouk University", "year": "2024"}],
        "languages": [{"language": "Arabic", "level": "native"},
                      {"language": "English", "level": "intermediate"}],
    })


def weak():
    return _record("cv_weak.txt", "Rana Mansour", "rana.mansour.test@example.com", {
        "name": "Rana Mansour",
        "skills": ["Adobe Photoshop", "Adobe Illustrator", "Figma", "Branding",
                   "Typography", "Social media content"],
        "work_experience": [
            {"job_title": "Graphic Designer", "company": "Bright Pixel Agency",
             "start_date": "Feb 2021", "end_date": "Present",
             "description": "Design marketing material and brand identities for retail clients."},
            {"job_title": "Junior Designer", "company": "Studio Noor",
             "start_date": "Jun 2019", "end_date": "Jan 2021",
             "description": "Prepared print layouts and social media posts."},
        ],
        "stated_years_experience": 5,
        "education": [{"degree": "Diploma", "field": "Graphic Design",
                       "institution": "Al-Balqa Applied University", "year": "2019"}],
        "languages": [{"language": "Arabic", "level": "native"}],
    })


def unreadable():
    """A CV whose text could not be extracted (for example a scanned image)."""
    record = _record("cv_weak.txt", "Scan Only", "scan.only.test@example.com", {})
    record.update(cv_text="", cv_info=None, original_filename="scanned_cv.pdf",
                  extraction={"status": "empty"})
    return record
