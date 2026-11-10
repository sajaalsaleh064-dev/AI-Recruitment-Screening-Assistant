"""Step 5 - Turn a CV file (PDF or DOCX) into clean text.

What this module does:
1. Reads the text out of a PDF (pypdf) or a DOCX (python-docx).
2. Cleans it: fixes odd characters, removes extra spaces and blank lines.
3. Reports clearly when a file is empty or cannot be read.
4. Stores the text together with basic candidate information as JSON.

Nothing is guessed here. The candidate's name and address come from the
email's "From" line; emails and phone numbers are only copied from the CV
when they literally appear in it.
"""
import json
import logging
import os
import re
import unicodedata
from datetime import datetime, timezone
from email.utils import parseaddr

from docx import Document
from docx.table import Table
from pypdf import PdfReader

from src.config import EXTRACTED_DIR, MIN_CV_TEXT_CHARS

logger = logging.getLogger(__name__)

BULLET_CHARS = "•●▪■◦‣○►➢❖✓✔⦁∙"
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"(?<![\w/.-])(?:\+|00)?\d[\d\s().-]{6,}\d(?![\w/-])")


# ---------------------------------------------------------------------------
# 1. Reading the files
# ---------------------------------------------------------------------------
def extract_pdf_text(path):
    """Return (text, page_count) for a PDF file.

    pypdf reads the text layer of each page. A scanned PDF is only a
    picture of text, so it has no text layer and returns an empty string.
    """
    reader = PdfReader(path)
    if reader.is_encrypted:
        # Many "protected" PDFs open with an empty password.
        if not reader.decrypt(""):
            raise ValueError("PDF is password-protected")
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages), len(pages)


def extract_docx_text(path):
    """Return the text of a DOCX file, including text inside tables.

    Many CV templates put the whole layout in tables, so reading only the
    paragraphs would miss most of the content. iter_inner_content() walks
    paragraphs and tables in the order they appear in the document.
    """
    document = Document(path)
    lines = []
    for block in document.iter_inner_content():
        if isinstance(block, Table):
            for row in block.rows:
                cells = []
                for cell in row.cells:
                    text = cell.text.strip()
                    # Merged cells are returned more than once: keep one copy.
                    if text and (not cells or cells[-1] != text):
                        cells.append(text)
                if cells:
                    lines.append(" | ".join(cells))
        else:
            lines.append(block.text)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------
def clean_text(text):
    """Normalize raw CV text so later steps get consistent input."""
    if not text:
        return ""
    # NFKC turns look-alike characters into normal ones: the single-character
    # ligature "ﬁ" becomes "fi", a non-breaking space becomes a space, etc.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    # Remove invisible characters (control codes, zero-width, soft hyphen).
    text = re.sub(r"[\x00-\x09\x0b-\x1f\x7f­​-‏﻿]", "", text)
    # Different bullet symbols all become a simple dash.
    text = re.sub(f"[{re.escape(BULLET_CHARS)}]", "- ", text)
    text = re.sub(r" {2,}", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line in a row
    return text.strip()


# ---------------------------------------------------------------------------
# 3. One file -> one result
# ---------------------------------------------------------------------------
def extract_cv_text(path, file_type=None):
    """Extract and clean the text of one CV file.

    Always returns a dictionary and never raises, so one bad CV cannot
    stop the others:
        status "ok"     -> text is ready
        status "empty"  -> file opened but has (almost) no text
        status "failed" -> file is missing, corrupted or unsupported
    """
    result = {"status": "failed", "reason": "", "text": "", "char_count": 0,
              "page_count": None}
    file_type = file_type or os.path.splitext(path)[1].lower().lstrip(".")

    if not os.path.exists(path):
        result["reason"] = "File not found"
        return result

    try:
        if file_type == "pdf":
            raw, result["page_count"] = extract_pdf_text(path)
        elif file_type == "docx":
            raw = extract_docx_text(path)
        else:
            result["reason"] = f"Unsupported file type: {file_type or 'unknown'}"
            return result
    except Exception as e:  # pypdf / python-docx raise many different errors
        result["reason"] = f"Could not read file: {type(e).__name__}: {e}"
        logger.warning("%s: %s", path, result["reason"])
        return result

    text = clean_text(raw)
    result["text"] = text
    result["char_count"] = len(text)
    if len(text) < MIN_CV_TEXT_CHARS:
        result["status"] = "empty"
        result["reason"] = (
            f"Only {len(text)} characters of text found. The file may be a "
            "scanned image or blank. Needs manual review by HR."
        )
        logger.warning("%s: %s", path, result["reason"])
    else:
        result["status"] = "ok"
        result["reason"] = f"Extracted {len(text)} characters"
    return result


# ---------------------------------------------------------------------------
# 4. Basic candidate information + storage
# ---------------------------------------------------------------------------
def find_contact_details(text):
    """Return emails and phone numbers that literally appear in the CV text."""
    emails = sorted(set(EMAIL_PATTERN.findall(text)), key=str.lower)
    phones = []
    for match in PHONE_PATTERN.findall(text):
        digits = re.sub(r"\D", "", match)
        # Real phone numbers have 8-15 digits; this skips years like 2019-2023.
        if 8 <= len(digits) <= 15 and not re.fullmatch(r"\d{4}\s*-\s*\d{4}", match.strip()):
            phones.append(match.strip())
    return emails, sorted(set(phones))


def build_cv_record(download_result):
    """Combine email details, extraction result and CV text in one record.

    `download_result` is one "saved" result from Step 4
    (attachment_downloader.save_cv_attachments).
    """
    extraction = extract_cv_text(download_result["path"], download_result["file_type"])
    sender_name, sender_email = parseaddr(download_result.get("sender", ""))
    cv_emails, cv_phones = find_contact_details(extraction["text"])
    return {
        "message_id": download_result["message_id"],
        "candidate": {
            "name_from_email": sender_name,
            "email_from_email": sender_email,
            "emails_in_cv": cv_emails,
            "phones_in_cv": cv_phones,
        },
        "email_subject": download_result.get("subject", ""),
        "cv_file": download_result["path"],
        "original_filename": download_result.get("filename", ""),
        "file_type": download_result["file_type"],
        "extraction": {
            "status": extraction["status"],
            "reason": extraction["reason"],
            "char_count": extraction["char_count"],
            "page_count": extraction["page_count"],
        },
        "cv_text": extraction["text"],
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def save_cv_record(record, dest_dir=EXTRACTED_DIR):
    """Write one record as a JSON file and return its path."""
    os.makedirs(dest_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(record["cv_file"]))[0]
    path = os.path.join(dest_dir, f"{stem}.json")
    with open(path, "w", encoding="utf-8") as f:
        # ensure_ascii=False keeps Arabic and accented letters readable.
        json.dump(record, f, indent=2, ensure_ascii=False)
    return path


def load_cv_records(source_dir=EXTRACTED_DIR):
    """Read every stored record (used by the AI steps later)."""
    records = []
    if not os.path.isdir(source_dir):
        return records
    for name in sorted(os.listdir(source_dir)):
        if name.endswith(".json"):
            with open(os.path.join(source_dir, name), "r", encoding="utf-8") as f:
                records.append(json.load(f))
    return records


if __name__ == "__main__":
    # Quick manual check on one file:  python -m src.text_extractor path/to/cv.pdf
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.text_extractor <path to a .pdf or .docx file>")
        sys.exit(1)
    outcome = extract_cv_text(sys.argv[1])
    print(f"Status : {outcome['status']}")
    print(f"Reason : {outcome['reason']}")
    print(f"Pages  : {outcome['page_count']}")
    print("-" * 60)
    print(outcome["text"])
