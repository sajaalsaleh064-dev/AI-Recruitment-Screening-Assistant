"""Loads settings from the .env file so nothing is hard-coded in the code."""
import os
from dotenv import load_dotenv

load_dotenv()

CREDENTIALS_PATH = os.getenv("CREDENTIALS_PATH", "credentials.json")
TOKEN_PATH = os.getenv("TOKEN_PATH", "token.json")
GMAIL_QUERY = os.getenv(
    "GMAIL_QUERY",
    "has:attachment (filename:pdf OR filename:docx) subject:application",
)
# File that remembers which emails were already processed (no duplicates).
PROCESSED_IDS_PATH = os.getenv("PROCESSED_IDS_PATH", "data/processed_ids.json")
# Folder where downloaded CV files are kept temporarily (ignored by git).
CV_DOWNLOAD_DIR = os.getenv("CV_DOWNLOAD_DIR", "data/cvs")
# Attachments bigger than this are skipped (a CV is never this large).
MAX_ATTACHMENT_MB = int(os.getenv("MAX_ATTACHMENT_MB", "10"))
# Folder where the extracted CV text is stored as JSON (ignored by git).
EXTRACTED_DIR = os.getenv("EXTRACTED_DIR", "data/output/extracted")
# A CV with less text than this is treated as empty (probably a scanned image).
MIN_CV_TEXT_CHARS = int(os.getenv("MIN_CV_TEXT_CHARS", "50"))
# The job description (free text) and its structured list of requirements.
JOB_DESCRIPTION_PATH = os.getenv("JOB_DESCRIPTION_PATH", "job/job_description.txt")
JOB_REQUIREMENTS_PATH = os.getenv("JOB_REQUIREMENTS_PATH", "job/job_requirements.json")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
# Address of the local Ollama server.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# Longer texts are cut to this length before they are sent to the LLM.
MAX_LLM_INPUT_CHARS = int(os.getenv("MAX_LLM_INPUT_CHARS", "12000"))
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
# Cosine similarity limits used in matching (0 = unrelated, 1 = same meaning).
# At or above MEETS -> "Meets"; at or above PARTIAL -> "Partially Meets".
SIMILARITY_MEETS = float(os.getenv("SIMILARITY_MEETS", "0.55"))
SIMILARITY_PARTIAL = float(os.getenv("SIMILARITY_PARTIAL", "0.35"))
# A past job counts as related to the required field at or above this value.
SIMILARITY_RELEVANT = float(os.getenv("SIMILARITY_RELEVANT", "0.40"))
# In the overall score a Must-Have counts this many times an optional requirement.
MUST_HAVE_WEIGHT = float(os.getenv("MUST_HAVE_WEIGHT", "2"))
# Where the final screening report is saved (ignored by git).
REPORT_PATH = os.getenv("REPORT_PATH", "data/output/screening_report.json")

# Read-only access to Gmail: the app can read emails but never send or delete.
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
