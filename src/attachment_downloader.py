"""Step 4 - Download CV attachments (PDF / DOCX) from application emails.

What this module does:
1. Skips emails that have no attachments.
2. Works out the file type from the extension and the MIME type.
3. Ignores anything that is not a PDF or DOCX.
4. Downloads the file with the Gmail Attachments API and decodes the Base64.
5. Checks that the content really is a PDF / DOCX (not corrupted or renamed).
6. Saves the file in a temporary folder, ready for text extraction (Step 5).

One bad attachment never stops the others: every file gets its own result
with a status ("saved", "skipped" or "failed") and a reason.
"""
import base64
import binascii
import io
import logging
import os
import re
import zipfile

from googleapiclient.errors import HttpError

from src.config import CV_DOWNLOAD_DIR, MAX_ATTACHMENT_MB

logger = logging.getLogger(__name__)

PDF_MIME = "application/pdf"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


# ---------------------------------------------------------------------------
# 1. File type
# ---------------------------------------------------------------------------
def detect_file_type(filename, mime_type):
    """Return "pdf", "docx" or None (unsupported).

    The extension is checked first. If the file has no useful extension,
    the MIME type decides. Some mail apps send every attachment as
    "application/octet-stream", which is why the extension comes first.
    """
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return "pdf"
    if name.endswith(".docx"):
        return "docx"
    if mime_type == PDF_MIME:
        return "pdf"
    if mime_type == DOCX_MIME:
        return "docx"
    return None


def validate_content(data, file_type):
    """Check that the downloaded bytes really are the claimed file type.

    Returns (True/False, reason). This catches empty, truncated or renamed
    files (for example a .exe renamed to .pdf) before they are saved.
    """
    if not data:
        return False, "File is empty"
    if len(data) > MAX_ATTACHMENT_MB * 1024 * 1024:
        return False, f"File is larger than {MAX_ATTACHMENT_MB} MB"

    if file_type == "pdf":
        # Every PDF starts with the text "%PDF-" (allow a little leading junk).
        if b"%PDF-" not in data[:1024]:
            return False, "Not a real PDF (missing %PDF header)"
        return True, "Valid PDF header"

    if file_type == "docx":
        # A DOCX is a ZIP archive that contains word/document.xml.
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                if "word/document.xml" not in archive.namelist():
                    return False, "Not a real DOCX (word/document.xml missing)"
                if archive.testzip() is not None:
                    return False, "DOCX archive is corrupted"
        except zipfile.BadZipFile:
            return False, "Not a real DOCX (not a valid ZIP archive)"
        return True, "Valid DOCX archive"

    return False, "Unsupported file type"


# ---------------------------------------------------------------------------
# 2. Downloading
# ---------------------------------------------------------------------------
def _decode_base64url(data):
    """Turn Gmail's URL-safe Base64 text back into the original bytes."""
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def download_attachment(service, message_id, attachment):
    """Return the raw bytes of one attachment.

    Large files are fetched with the Attachments API using attachment_id.
    Very small files may already be inside the message (inline_data).
    """
    encoded = attachment.get("inline_data")
    if not encoded:
        if not attachment.get("attachment_id"):
            raise ValueError("Attachment has no ID and no inline data")
        response = (
            service.users()
            .messages()
            .attachments()
            .get(userId="me", messageId=message_id, id=attachment["attachment_id"])
            .execute()
        )
        encoded = response.get("data", "")
    return _decode_base64url(encoded)


# ---------------------------------------------------------------------------
# 3. Saving
# ---------------------------------------------------------------------------
def safe_filename(message_id, filename, file_type):
    """Build a safe, unique file name: <message id>_<clean name>.<type>.

    The sender controls the attachment name, so it must never be trusted:
    folder parts ("../") and unusual characters are removed. The message ID
    prefix stops two candidates' "CV.pdf" from overwriting each other.
    """
    base = os.path.basename((filename or "").replace("\\", "/"))
    stem = os.path.splitext(base)[0]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")[:60] or "cv"
    return f"{message_id}_{stem}.{file_type}"


def _unique_path(folder, name):
    """Avoid overwriting when one email has two files with the same name."""
    path = os.path.join(folder, name)
    stem, ext = os.path.splitext(name)
    counter = 2
    while os.path.exists(path):
        path = os.path.join(folder, f"{stem}_{counter}{ext}")
        counter += 1
    return path


def save_cv_attachments(service, email, dest_dir=CV_DOWNLOAD_DIR):
    """Download and save every PDF/DOCX attachment of one email.

    Returns one result dictionary per attachment:
        status "saved"   -> file is on disk at result["path"]
        status "skipped" -> unsupported file type (ignored on purpose)
        status "failed"  -> download error or corrupted file
    An email without attachments returns an empty list.
    """
    results = []
    attachments = email.get("attachments", [])
    if not attachments:
        logger.info("Email %s has no attachments, ignoring", email.get("message_id"))
        return results

    for attachment in attachments:
        result = {
            "message_id": email["message_id"],
            "sender": email.get("sender", ""),
            "subject": email.get("subject", ""),
            "filename": attachment["filename"],
            "file_type": detect_file_type(attachment["filename"], attachment["mime_type"]),
            "path": None,
            "status": "failed",
            "reason": "",
        }
        results.append(result)

        if result["file_type"] is None:
            result["status"] = "skipped"
            result["reason"] = f"Unsupported file type ({attachment['mime_type']})"
            continue

        try:
            data = download_attachment(service, email["message_id"], attachment)
        except (HttpError, ValueError, binascii.Error) as e:
            result["reason"] = f"Download failed: {e}"
            logger.error("%s: %s", attachment["filename"], result["reason"])
            continue

        valid, reason = validate_content(data, result["file_type"])
        if not valid:
            result["reason"] = reason
            logger.warning("%s rejected: %s", attachment["filename"], reason)
            continue

        os.makedirs(dest_dir, exist_ok=True)
        path = _unique_path(
            dest_dir,
            safe_filename(email["message_id"], attachment["filename"], result["file_type"]),
        )
        try:
            with open(path, "wb") as f:  # "wb" = write bytes, not text
                f.write(data)
        except OSError as e:
            result["reason"] = f"Could not save file: {e}"
            logger.error("%s: %s", attachment["filename"], result["reason"])
            continue

        result.update(status="saved", path=path, reason=f"{reason}, {len(data)} bytes")
        logger.info("Saved %s", path)

    return results


def download_cvs(service, emails, dest_dir=CV_DOWNLOAD_DIR):
    """Run save_cv_attachments for a list of emails and join the results."""
    results = []
    for email in emails:
        results.extend(save_cv_attachments(service, email, dest_dir))
    return results


if __name__ == "__main__":
    from src.email_processor import fetch_new_application_emails
    from src.gmail_client import get_gmail_service

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    gmail = get_gmail_service()
    found = download_cvs(gmail, fetch_new_application_emails(gmail, max_results=10))
    if not found:
        print("No attachments to download.")
    for r in found:
        print(f"\n[{r['status'].upper()}] {r['filename']}")
        print(f"  From   : {r['sender']}")
        print(f"  Type   : {r['file_type']}")
        print(f"  Reason : {r['reason']}")
        print(f"  Path   : {r['path']}")
