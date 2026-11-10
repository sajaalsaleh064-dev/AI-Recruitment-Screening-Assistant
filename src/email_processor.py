"""Step 3 - Find job-application emails and read their details.

What this module does:
1. Search Gmail with a query (for example: has:attachment filename:pdf).
2. Download each matching message and pull out the useful parts:
   sender, subject, message ID, body text and attachment information.
3. Decide whether the email looks like a job application.
4. Remember which message IDs were already handled, so the same email
   is never processed twice.

Everything here only READS from Gmail. Nothing is sent, deleted or modified.
"""
import base64
import html
import json
import logging
import os
import re

from googleapiclient.errors import HttpError

from src.config import GMAIL_QUERY, PROCESSED_IDS_PATH

logger = logging.getLogger(__name__)

# CV file types we accept, by extension and by MIME type.
SUPPORTED_EXTENSIONS = (".pdf", ".docx")
SUPPORTED_MIME_TYPES = (
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
)

# Words that suggest the email is a job application.
APPLICATION_KEYWORDS = (
    "application", "applying", "apply", "cv", "resume", "résumé",
    "curriculum vitae", "position", "vacancy", "job",
)


# ---------------------------------------------------------------------------
# 1. Searching
# ---------------------------------------------------------------------------
def search_message_ids(service, query=GMAIL_QUERY, max_results=50):
    """Return the IDs of messages that match a Gmail search query.

    Gmail returns results in pages, so we keep asking for the next page
    until we have `max_results` IDs or there are no more pages.
    """
    ids = []
    page_token = None
    try:
        while len(ids) < max_results:
            response = (
                service.users()
                .messages()
                .list(
                    userId="me",
                    q=query,
                    maxResults=min(100, max_results - len(ids)),
                    pageToken=page_token,
                )
                .execute()
            )
            ids.extend(m["id"] for m in response.get("messages", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break
    except HttpError as e:
        logger.error("Gmail search failed for query %r: %s", query, e)
        raise
    logger.info("Query %r matched %d message(s)", query, len(ids))
    return ids[:max_results]


def get_message(service, message_id):
    """Download one full message (headers, body and attachment list)."""
    return (
        service.users()
        .messages()
        .get(userId="me", id=message_id, format="full")
        .execute()
    )


# ---------------------------------------------------------------------------
# 2. Parsing
# ---------------------------------------------------------------------------
def _decode_base64url(data):
    """Gmail sends content as URL-safe Base64 text. Turn it back into bytes."""
    if not data:
        return b""
    padding = "=" * (-len(data) % 4)  # Base64 length must be a multiple of 4
    return base64.urlsafe_b64decode(data + padding)


def _get_header(headers, name):
    """Find one header (such as From or Subject), ignoring upper/lower case."""
    for header in headers:
        if header.get("name", "").lower() == name.lower():
            return header.get("value", "")
    return ""


def _walk_parts(payload):
    """Yield every part of a message, including parts nested inside parts.

    An email is a tree: a "multipart" part contains other parts
    (plain text, HTML, attachments...). This visits all of them.
    """
    yield payload
    for part in payload.get("parts", []) or []:
        yield from _walk_parts(part)


def _html_to_text(raw_html):
    """Very small HTML-to-text conversion for emails that have no plain text."""
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw_html)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(text)


def _extract_body(payload):
    """Return the email body as text. Plain text is preferred over HTML."""
    plain, html_parts = [], []
    for part in _walk_parts(payload):
        if part.get("filename"):  # this part is an attachment, not the body
            continue
        data = part.get("body", {}).get("data")
        if not data:
            continue
        text = _decode_base64url(data).decode("utf-8", errors="replace")
        if part.get("mimeType") == "text/plain":
            plain.append(text)
        elif part.get("mimeType") == "text/html":
            html_parts.append(_html_to_text(text))
    body = "\n".join(plain) if plain else "\n".join(html_parts)
    # Tidy up: collapse repeated spaces and blank lines.
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n\s*\n+", "\n\n", body)
    return body.strip()


def is_supported_cv_file(filename, mime_type):
    """True if the file is a PDF or DOCX, judged by extension or MIME type."""
    name = (filename or "").lower()
    return name.endswith(SUPPORTED_EXTENSIONS) or mime_type in SUPPORTED_MIME_TYPES


def _extract_attachments(payload):
    """List the attachments of a message (information only, no download)."""
    attachments = []
    for part in _walk_parts(payload):
        filename = part.get("filename")
        if not filename:
            continue
        body = part.get("body", {})
        mime_type = part.get("mimeType", "")
        attachments.append(
            {
                "filename": filename,
                "mime_type": mime_type,
                "size": body.get("size", 0),
                # Needed in Step 4 to download the file itself.
                "attachment_id": body.get("attachmentId"),
                # Very small files can arrive inside the message itself.
                "inline_data": body.get("data"),
                "supported": is_supported_cv_file(filename, mime_type),
            }
        )
    return attachments


def parse_message(message):
    """Turn a raw Gmail message into a simple, clean dictionary."""
    payload = message.get("payload", {})
    headers = payload.get("headers", [])
    return {
        "message_id": message.get("id", ""),
        "thread_id": message.get("threadId", ""),
        "sender": _get_header(headers, "From"),
        "subject": _get_header(headers, "Subject"),
        "date": _get_header(headers, "Date"),
        "body": _extract_body(payload),
        "attachments": _extract_attachments(payload),
    }


# ---------------------------------------------------------------------------
# 3. Is this a job application?
# ---------------------------------------------------------------------------
def is_application_email(email):
    """Decide whether an email looks like a job application.

    Returns (True/False, reason) so the decision is always explainable.
    Rule: it must carry at least one PDF/DOCX file AND mention an
    application-related word in the subject, body or file name.
    """
    cv_files = [a for a in email["attachments"] if a["supported"]]
    if not cv_files:
        return False, "No PDF or DOCX attachment"

    searchable = " ".join(
        [email["subject"], email["body"]] + [a["filename"] for a in cv_files]
    ).lower()
    for keyword in APPLICATION_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", searchable):
            return True, f"Has CV file and mentions '{keyword}'"
    return False, "Has PDF/DOCX file but no application-related wording"


# ---------------------------------------------------------------------------
# 4. Tracking processed messages (no duplicates)
# ---------------------------------------------------------------------------
def load_processed_ids(path=PROCESSED_IDS_PATH):
    """Read the set of message IDs that were already processed."""
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Could not read %s (%s). Starting with an empty list.", path, e)
        return set()


def save_processed_ids(ids, path=PROCESSED_IDS_PATH):
    """Write the set of processed message IDs to disk."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(ids), f, indent=2)


def mark_processed(message_id, path=PROCESSED_IDS_PATH):
    """Record one message as processed. Call this after its CV is handled."""
    ids = load_processed_ids(path)
    ids.add(message_id)
    save_processed_ids(ids, path)


# ---------------------------------------------------------------------------
# 5. Putting it together
# ---------------------------------------------------------------------------
def fetch_new_application_emails(
    service, query=GMAIL_QUERY, max_results=50, processed_path=PROCESSED_IDS_PATH
):
    """Return new application emails that were not processed before.

    This function does NOT mark emails as processed. That happens later
    (mark_processed) once the CV has really been handled, so a crash in a
    later step never causes an application to be silently skipped.
    """
    already_done = load_processed_ids(processed_path)
    applications = []

    for message_id in search_message_ids(service, query, max_results):
        if message_id in already_done:
            logger.info("Skipping %s: already processed", message_id)
            continue
        try:
            email = parse_message(get_message(service, message_id))
        except HttpError as e:
            logger.error("Could not read message %s: %s", message_id, e)
            continue

        relevant, reason = is_application_email(email)
        email["relevance_reason"] = reason
        if relevant:
            applications.append(email)
        else:
            logger.info("Ignoring %s (%r): %s", message_id, email["subject"], reason)

    logger.info("Found %d new application email(s)", len(applications))
    return applications


def _print_summary(emails):
    """Show the emails in a readable form (body shortened for privacy)."""
    if not emails:
        print("No new application emails found.")
        return
    for i, email in enumerate(emails, start=1):
        print(f"\n--- Application {i} ---")
        print(f"Message ID : {email['message_id']}")
        print(f"From       : {email['sender']}")
        print(f"Subject    : {email['subject']}")
        print(f"Date       : {email['date']}")
        print(f"Why        : {email['relevance_reason']}")
        print(f"Body       : {email['body'][:150]!r}")
        for a in email["attachments"]:
            status = "CV file" if a["supported"] else "ignored (unsupported type)"
            print(f"Attachment : {a['filename']} | {a['mime_type']} | "
                  f"{a['size']} bytes | {status}")


if __name__ == "__main__":
    from src.gmail_client import get_gmail_service

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    print(f"Searching Gmail with: {GMAIL_QUERY}")
    _print_summary(fetch_new_application_emails(get_gmail_service(), max_results=10))
