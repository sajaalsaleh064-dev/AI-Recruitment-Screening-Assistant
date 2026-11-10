"""Step 4 tests: downloading and validating CV attachments.

Uses a FAKE Gmail service and files built in memory, so no internet,
no credentials and no real CVs are needed.

Run:  python -m unittest tests.test_attachments -v
"""
import base64
import io
import os
import tempfile
import unittest

from docx import Document
from pypdf import PdfWriter

from src import attachment_downloader as ad

DOCX_MIME = ad.DOCX_MIME


def make_pdf_bytes():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def make_docx_bytes(text="Python developer with 3 years of experience"):
    document = Document()
    document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def b64(data):
    """Encode bytes the way Gmail does (URL-safe Base64, no padding)."""
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


class FakeRequest:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class FakeGmailService:
    """Imitates service.users().messages().attachments().get()."""

    def __init__(self, files):
        self._files = files  # {attachment_id: bytes}
        self.requested = []

    def users(self):
        return self

    def messages(self):
        return self

    def attachments(self):
        return self

    def get(self, userId, messageId, id):
        self.requested.append(id)
        return FakeRequest({"data": b64(self._files[id]), "size": len(self._files[id])})


def attachment(filename, mime, attachment_id=None, inline=None):
    return {"filename": filename, "mime_type": mime, "size": 0,
            "attachment_id": attachment_id, "inline_data": inline, "supported": True}


def email(message_id, attachments):
    return {"message_id": message_id, "sender": "Sara <sara@example.com>",
            "subject": "Application", "attachments": attachments}


class DetectFileTypeTests(unittest.TestCase):
    def test_by_extension(self):
        self.assertEqual(ad.detect_file_type("cv.PDF", "application/octet-stream"), "pdf")
        self.assertEqual(ad.detect_file_type("cv.docx", "application/octet-stream"), "docx")

    def test_by_mime_type_when_extension_is_missing(self):
        self.assertEqual(ad.detect_file_type("my_cv", "application/pdf"), "pdf")
        self.assertEqual(ad.detect_file_type("my_cv", DOCX_MIME), "docx")

    def test_unsupported_types(self):
        self.assertIsNone(ad.detect_file_type("photo.png", "image/png"))
        self.assertIsNone(ad.detect_file_type("old_cv.doc", "application/msword"))
        self.assertIsNone(ad.detect_file_type("cv.zip", "application/zip"))


class ValidateContentTests(unittest.TestCase):
    def test_real_pdf_and_docx_are_valid(self):
        self.assertTrue(ad.validate_content(make_pdf_bytes(), "pdf")[0])
        self.assertTrue(ad.validate_content(make_docx_bytes(), "docx")[0])

    def test_empty_file_is_rejected(self):
        self.assertEqual(ad.validate_content(b"", "pdf"), (False, "File is empty"))

    def test_fake_pdf_is_rejected(self):
        valid, reason = ad.validate_content(b"MZ this is really a program", "pdf")
        self.assertFalse(valid)
        self.assertIn("Not a real PDF", reason)

    def test_corrupted_docx_is_rejected(self):
        valid, reason = ad.validate_content(make_docx_bytes()[:200], "docx")
        self.assertFalse(valid)
        self.assertIn("Not a real DOCX", reason)

    def test_zip_that_is_not_a_docx_is_rejected(self):
        import zipfile
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("hello.txt", "hi")
        valid, reason = ad.validate_content(buffer.getvalue(), "docx")
        self.assertFalse(valid)
        self.assertIn("word/document.xml missing", reason)

    def test_too_large_file_is_rejected(self):
        big = b"%PDF-" + b"0" * (ad.MAX_ATTACHMENT_MB * 1024 * 1024 + 1)
        valid, reason = ad.validate_content(big, "pdf")
        self.assertFalse(valid)
        self.assertIn("larger than", reason)


class SafeFilenameTests(unittest.TestCase):
    def test_path_traversal_is_removed(self):
        name = ad.safe_filename("m1", "../../Windows/evil.pdf", "pdf")
        self.assertEqual(name, "m1_evil.pdf")

    def test_strange_characters_are_replaced(self):
        self.assertEqual(ad.safe_filename("m1", "My CV (final)!.docx", "docx"),
                         "m1_My_CV_final.docx")

    def test_empty_name_gets_a_default(self):
        self.assertEqual(ad.safe_filename("m1", "", "pdf"), "m1_cv.pdf")


class SaveAttachmentsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = os.path.join(self.tmp.name, "cvs")
        self.pdf, self.docx = make_pdf_bytes(), make_docx_bytes()

    def tearDown(self):
        self.tmp.cleanup()

    def test_email_without_attachments_is_ignored(self):
        self.assertEqual(ad.save_cv_attachments(FakeGmailService({}), email("m0", []), self.dest), [])
        self.assertFalse(os.path.exists(self.dest))

    def test_pdf_is_downloaded_decoded_and_saved(self):
        service = FakeGmailService({"a1": self.pdf})
        results = ad.save_cv_attachments(
            service, email("m1", [attachment("Sara_CV.pdf", "application/pdf", "a1")]), self.dest)
        self.assertEqual(results[0]["status"], "saved")
        self.assertEqual(os.path.basename(results[0]["path"]), "m1_Sara_CV.pdf")
        with open(results[0]["path"], "rb") as f:
            self.assertEqual(f.read(), self.pdf)  # identical to the original

    def test_docx_is_downloaded_and_saved(self):
        service = FakeGmailService({"a1": self.docx})
        results = ad.save_cv_attachments(
            service, email("m2", [attachment("omar.docx", DOCX_MIME, "a1")]), self.dest)
        self.assertEqual(results[0]["status"], "saved")
        self.assertEqual(results[0]["file_type"], "docx")
        self.assertEqual(Document(results[0]["path"]).paragraphs[0].text,
                         "Python developer with 3 years of experience")

    def test_multiple_attachments_mixed_results(self):
        service = FakeGmailService({"a1": self.pdf, "a2": self.docx, "a4": b"not a pdf"})
        results = ad.save_cv_attachments(service, email("m3", [
            attachment("cv.pdf", "application/pdf", "a1"),
            attachment("cover_letter.docx", DOCX_MIME, "a2"),
            attachment("photo.png", "image/png", "a3"),
            attachment("broken.pdf", "application/pdf", "a4"),
        ]), self.dest)
        self.assertEqual([r["status"] for r in results],
                         ["saved", "saved", "skipped", "failed"])
        self.assertNotIn("a3", service.requested)  # unsupported file never downloaded
        self.assertEqual(len(os.listdir(self.dest)), 2)  # broken file not saved

    def test_inline_data_is_used_without_calling_the_api(self):
        service = FakeGmailService({})
        results = ad.save_cv_attachments(service, email("m4", [
            attachment("small.pdf", "application/pdf", inline=b64(self.pdf))]), self.dest)
        self.assertEqual(results[0]["status"], "saved")
        self.assertEqual(service.requested, [])

    def test_same_name_twice_does_not_overwrite(self):
        service = FakeGmailService({"a1": self.pdf, "a2": self.pdf})
        results = ad.save_cv_attachments(service, email("m5", [
            attachment("cv.pdf", "application/pdf", "a1"),
            attachment("cv.pdf", "application/pdf", "a2"),
        ]), self.dest)
        self.assertNotEqual(results[0]["path"], results[1]["path"])
        self.assertEqual(len(os.listdir(self.dest)), 2)

    def test_download_error_is_reported_not_raised(self):
        results = ad.save_cv_attachments(FakeGmailService({}), email("m6", [
            attachment("cv.pdf", "application/pdf")]), self.dest)  # no ID, no data
        self.assertEqual(results[0]["status"], "failed")
        self.assertIn("Download failed", results[0]["reason"])


if __name__ == "__main__":
    unittest.main()
