"""Step 5 tests: CV text extraction, cleaning and record storage.

The PDF and DOCX files are created by the tests themselves in a temporary
folder, so no real CVs are needed.

Run:  python -m unittest tests.test_text_extractor -v
"""
import json
import os
import tempfile
import unittest

from docx import Document
from pypdf import PdfWriter

from src import text_extractor as te

CV_LINES = [
    "Sara Ali",
    "Email: sara.ali@example.com | Phone: +962 79 123 4567",
    "Python Developer with 3 years of experience (2021 - 2024)",
    "Skills: Python, Django, REST APIs, PostgreSQL, Git",
    "Education: BSc Computer Science, University of Jordan",
]


def write_text_pdf(path, lines):
    """Write a minimal one-page PDF containing the given lines of text."""
    def escape(s):
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    content = "BT /F1 11 Tf 50 780 Td 14 TL\n"
    content += "\n".join(f"({escape(line)}) Tj T*" for line in lines) + "\nET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref_at = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        pdf += f"{offset:010d} 00000 n \n".encode()
    pdf += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_at}\n%%EOF\n").encode()
    with open(path, "wb") as f:
        f.write(pdf)


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def path(self, name):
        return os.path.join(self.tmp.name, name)


class CleanTextTests(unittest.TestCase):
    def test_extra_spaces_and_blank_lines_are_removed(self):
        raw = "  Sara   Ali  \n\n\n\n  Python\t\tDeveloper  \r\n"
        self.assertEqual(te.clean_text(raw), "Sara Ali\n\nPython Developer")

    def test_bullets_become_dashes(self):
        self.assertEqual(te.clean_text("• Python\n●  Django"), "- Python\n- Django")

    def test_ligatures_and_special_spaces_are_normalized(self):
        self.assertEqual(te.clean_text("Certiﬁed engineer"), "Certified engineer")

    def test_invisible_characters_are_removed(self):
        self.assertEqual(te.clean_text("Py​thon\x00 dev­eloper"), "Python developer")

    def test_empty_input(self):
        self.assertEqual(te.clean_text(""), "")
        self.assertEqual(te.clean_text(None), "")


class PdfExtractionTests(TempDirTestCase):
    def test_text_is_extracted_from_pdf(self):
        write_text_pdf(self.path("cv.pdf"), CV_LINES)
        result = te.extract_cv_text(self.path("cv.pdf"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["page_count"], 1)
        self.assertIn("Sara Ali", result["text"])
        self.assertIn("Python, Django, REST APIs", result["text"])

    def test_pdf_without_text_is_reported_as_empty(self):
        writer = PdfWriter()
        writer.add_blank_page(width=595, height=842)
        with open(self.path("scan.pdf"), "wb") as f:
            writer.write(f)
        result = te.extract_cv_text(self.path("scan.pdf"))
        self.assertEqual(result["status"], "empty")
        self.assertIn("manual review", result["reason"])

    def test_corrupted_pdf_does_not_crash(self):
        with open(self.path("broken.pdf"), "wb") as f:
            f.write(b"%PDF-1.4 this is not really a pdf")
        result = te.extract_cv_text(self.path("broken.pdf"))
        self.assertEqual(result["status"], "failed")
        self.assertIn("Could not read file", result["reason"])


class DocxExtractionTests(TempDirTestCase):
    def test_paragraphs_and_tables_are_extracted_in_order(self):
        document = Document()
        document.add_paragraph("Omar Hassan")
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Skills"
        table.cell(0, 1).text = "Java, Spring Boot, SQL"
        table.cell(1, 0).text = "Languages"
        table.cell(1, 1).text = "Arabic, English"
        document.add_paragraph("Experience: 5 years as a backend engineer in Amman")
        document.save(self.path("cv.docx"))

        result = te.extract_cv_text(self.path("cv.docx"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["text"].split("\n"), [
            "Omar Hassan",
            "Skills | Java, Spring Boot, SQL",
            "Languages | Arabic, English",
            "Experience: 5 years as a backend engineer in Amman",
        ])

    def test_empty_docx_is_reported_as_empty(self):
        Document().save(self.path("blank.docx"))
        self.assertEqual(te.extract_cv_text(self.path("blank.docx"))["status"], "empty")

    def test_corrupted_docx_does_not_crash(self):
        with open(self.path("broken.docx"), "wb") as f:
            f.write(b"PK\x03\x04 broken")
        self.assertEqual(te.extract_cv_text(self.path("broken.docx"))["status"], "failed")


class OtherFileTests(TempDirTestCase):
    def test_missing_file(self):
        result = te.extract_cv_text(self.path("nothing.pdf"))
        self.assertEqual((result["status"], result["reason"]), ("failed", "File not found"))

    def test_unsupported_type(self):
        with open(self.path("cv.txt"), "w") as f:
            f.write("hello")
        result = te.extract_cv_text(self.path("cv.txt"))
        self.assertEqual(result["status"], "failed")
        self.assertIn("Unsupported file type", result["reason"])


class ContactDetailsTests(unittest.TestCase):
    def test_email_and_phone_are_found(self):
        emails, phones = te.find_contact_details("\n".join(CV_LINES))
        self.assertEqual(emails, ["sara.ali@example.com"])
        self.assertEqual(phones, ["+962 79 123 4567"])

    def test_year_ranges_are_not_phone_numbers(self):
        self.assertEqual(te.find_contact_details("Worked 2019 - 2023 and 2015-2018")[1], [])

    def test_nothing_is_invented_when_cv_has_no_contacts(self):
        self.assertEqual(te.find_contact_details("Python developer"), ([], []))


class RecordTests(TempDirTestCase):
    def download_result(self, name="m1_Sara_CV.pdf"):
        write_text_pdf(self.path(name), CV_LINES)
        return {"message_id": "m1", "sender": "Sara Ali <sara@example.com>",
                "subject": "Application for Python Developer", "filename": "Sara CV.pdf",
                "file_type": "pdf", "path": self.path(name), "status": "saved"}

    def test_record_contains_text_and_candidate_information(self):
        record = te.build_cv_record(self.download_result())
        self.assertEqual(record["message_id"], "m1")
        self.assertEqual(record["candidate"]["name_from_email"], "Sara Ali")
        self.assertEqual(record["candidate"]["email_from_email"], "sara@example.com")
        self.assertEqual(record["candidate"]["emails_in_cv"], ["sara.ali@example.com"])
        self.assertEqual(record["extraction"]["status"], "ok")
        self.assertIn("BSc Computer Science", record["cv_text"])

    def test_record_is_saved_and_loaded_again(self):
        record = te.build_cv_record(self.download_result())
        out_dir = self.path("extracted")
        saved_path = te.save_cv_record(record, out_dir)
        self.assertEqual(os.path.basename(saved_path), "m1_Sara_CV.json")
        with open(saved_path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["cv_text"], record["cv_text"])
        self.assertEqual(len(te.load_cv_records(out_dir)), 1)

    def test_unreadable_cv_still_gets_a_record_for_hr(self):
        result = self.download_result()
        with open(result["path"], "wb") as f:
            f.write(b"%PDF-1.4 broken")
        record = te.build_cv_record(result)
        self.assertEqual(record["extraction"]["status"], "failed")
        self.assertEqual(record["cv_text"], "")
        self.assertEqual(record["candidate"]["name_from_email"], "Sara Ali")


if __name__ == "__main__":
    unittest.main()
