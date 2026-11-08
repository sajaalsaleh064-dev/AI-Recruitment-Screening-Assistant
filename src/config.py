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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Read-only access to Gmail: the app can read emails but never send or delete.
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
