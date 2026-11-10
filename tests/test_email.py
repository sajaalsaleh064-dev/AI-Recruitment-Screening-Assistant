"""Step 3 tests: email parsing, filtering and duplicate tracking.

These tests use a FAKE Gmail service, so they need no internet,
no credentials.json and no real mailbox.

Run:  python -m unittest tests.test_email -v
"""
import base64
import os
import tempfile
import unittest

from src import email_processor as ep

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def b64(text):
    """Encode text the same way Gmail does (URL-safe Base64, no padding)."""
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode().rstrip("=")


def make_message(msg_id, sender, subject, body, attachments=(), html_only=False):
    """Build a fake message shaped like a real Gmail API response."""
    body_part = {
        "mimeType": "text/html" if html_only else "text/plain",
        "filename": "",
        "body": {"data": b64(body)},
    }
    parts = [{"mimeType": "multipart/alternative", "filename": "", "body": {},
              "parts": [body_part]}]
    for filename, mime in attachments:
        parts.append({
            "mimeType": mime,
            "filename": filename,
            "body": {"attachmentId": f"att-{filename}", "size": 1234},
        })
    return {
        "id": msg_id,
        "threadId": f"thread-{msg_id}",
        "payload": {
            "mimeType": "multipart/mixed",
            "headers": [
                {"name": "From", "value": sender},
                {"name": "subject", "value": subject},  # lower case on purpose
                {"name": "Date", "value": "Mon, 9 Nov 2026 10:00:00 +0000"},
            ],
            "parts": parts,
        },
    }


class FakeRequest:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class FakeGmailService:
    """Imitates service.users().messages().list() / .get()."""

    def __init__(self, messages, page_size=2):
        self._messages = {m["id"]: m for m in messages}
        self._page_size = page_size
        self.get_calls = []

    def users(self):
        return self

    def messages(self):
        return self

    def list(self, userId, q, maxResults, pageToken=None):
        ids = list(self._messages)
        start = int(pageToken or 0)
        end = start + min(self._page_size, maxResults)
        result = {"messages": [{"id": i} for i in ids[start:end]]}
        if end < len(ids):
            result["nextPageToken"] = str(end)
        return FakeRequest(result)

    def get(self, userId, id, format):
        self.get_calls.append(id)
        return FakeRequest(self._messages[id])


SAMPLE_MESSAGES = [
    make_message("m1", "Sara Ali <sara@example.com>", "Application for Python Developer",
                 "Dear HR,\n\nPlease find my CV attached.", [("Sara_CV.pdf", "application/pdf")]),
    make_message("m2", "Omar <omar@example.com>", "Applying for the open position",
                 "<p>Hello,</p><p>My r&eacute;sum&eacute; is attached.</p>",
                 [("omar.docx", DOCX_MIME)], html_only=True),
    make_message("m3", "Shop <shop@example.com>", "Your invoice",
                 "Thank you for your order.", [("invoice_991.pdf", "application/pdf")]),
    make_message("m4", "Lina <lina@example.com>", "Job application",
                 "My CV is attached.", [("photo.png", "image/png")]),
    make_message("m5", "Noor <noor@example.com>", "Application - Data Analyst",
                 "CV and certificate attached.",
                 [("noor_cv.PDF", "application/pdf"), ("logo.png", "image/png")]),
]


class ParseMessageTests(unittest.TestCase):
    def test_extracts_sender_subject_id_and_body(self):
        email = ep.parse_message(SAMPLE_MESSAGES[0])
        self.assertEqual(email["message_id"], "m1")
        self.assertEqual(email["sender"], "Sara Ali <sara@example.com>")
        self.assertEqual(email["subject"], "Application for Python Developer")
        self.assertIn("Please find my CV attached.", email["body"])

    def test_extracts_attachment_information(self):
        email = ep.parse_message(SAMPLE_MESSAGES[0])
        self.assertEqual(len(email["attachments"]), 1)
        attachment = email["attachments"][0]
        self.assertEqual(attachment["filename"], "Sara_CV.pdf")
        self.assertEqual(attachment["mime_type"], "application/pdf")
        self.assertEqual(attachment["attachment_id"], "att-Sara_CV.pdf")
        self.assertTrue(attachment["supported"])

    def test_html_only_body_is_converted_to_text(self):
        email = ep.parse_message(SAMPLE_MESSAGES[1])
        self.assertNotIn("<p>", email["body"])
        self.assertIn("résumé is attached", email["body"])

    def test_multiple_attachments_are_all_listed(self):
        email = ep.parse_message(SAMPLE_MESSAGES[4])
        supported = [a["supported"] for a in email["attachments"]]
        self.assertEqual(supported, [True, False])  # PDF yes, PNG no

    def test_message_without_parts_does_not_crash(self):
        email = ep.parse_message({"id": "x", "payload": {"headers": []}})
        self.assertEqual(email["body"], "")
        self.assertEqual(email["attachments"], [])


class FileTypeTests(unittest.TestCase):
    def test_pdf_and_docx_are_supported(self):
        self.assertTrue(ep.is_supported_cv_file("cv.pdf", "application/pdf"))
        self.assertTrue(ep.is_supported_cv_file("cv.DOCX", DOCX_MIME))

    def test_mime_type_is_enough_when_extension_is_missing(self):
        self.assertTrue(ep.is_supported_cv_file("my_cv", "application/pdf"))

    def test_other_types_are_not_supported(self):
        self.assertFalse(ep.is_supported_cv_file("photo.png", "image/png"))
        self.assertFalse(ep.is_supported_cv_file("cv.doc", "application/msword"))


class RelevanceTests(unittest.TestCase):
    def check(self, index):
        return ep.is_application_email(ep.parse_message(SAMPLE_MESSAGES[index]))

    def test_application_with_cv_is_relevant(self):
        relevant, reason = self.check(0)
        self.assertTrue(relevant)
        self.assertIn("application", reason)

    def test_invoice_pdf_is_not_an_application(self):
        relevant, reason = self.check(2)
        self.assertFalse(relevant)
        self.assertIn("no application-related wording", reason)

    def test_application_without_cv_file_is_ignored(self):
        relevant, reason = self.check(3)
        self.assertFalse(relevant)
        self.assertEqual(reason, "No PDF or DOCX attachment")


class ProcessedIdsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "data", "processed_ids.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_means_nothing_processed(self):
        self.assertEqual(ep.load_processed_ids(self.path), set())

    def test_mark_processed_is_saved_and_has_no_duplicates(self):
        ep.mark_processed("m1", self.path)
        ep.mark_processed("m1", self.path)
        ep.mark_processed("m2", self.path)
        self.assertEqual(ep.load_processed_ids(self.path), {"m1", "m2"})

    def test_corrupted_file_is_handled(self):
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "w") as f:
            f.write("{not valid json")
        self.assertEqual(ep.load_processed_ids(self.path), set())


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "processed_ids.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_search_follows_pages_and_respects_limit(self):
        service = FakeGmailService(SAMPLE_MESSAGES, page_size=2)
        self.assertEqual(ep.search_message_ids(service, "q"), ["m1", "m2", "m3", "m4", "m5"])
        self.assertEqual(ep.search_message_ids(service, "q", max_results=3), ["m1", "m2", "m3"])

    def test_only_real_applications_are_returned(self):
        service = FakeGmailService(SAMPLE_MESSAGES)
        emails = ep.fetch_new_application_emails(service, "q", processed_path=self.path)
        self.assertEqual([e["message_id"] for e in emails], ["m1", "m2", "m5"])

    def test_processed_emails_are_skipped_and_not_downloaded_again(self):
        ep.mark_processed("m1", self.path)
        service = FakeGmailService(SAMPLE_MESSAGES)
        emails = ep.fetch_new_application_emails(service, "q", processed_path=self.path)
        self.assertEqual([e["message_id"] for e in emails], ["m2", "m5"])
        self.assertNotIn("m1", service.get_calls)


if __name__ == "__main__":
    unittest.main()
