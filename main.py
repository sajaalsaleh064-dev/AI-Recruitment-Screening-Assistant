"""AI Recruitment Screening Assistant - main program.

Full workflow (Steps 2-9):
    Gmail -> application email -> CV attachment -> CV file -> extracted CV text
          -> structured candidate information (local LLM through Ollama)
          -> comparison with the job requirements (rules + embeddings)
          -> Must-Have filtering and ranking -> report for HR

The report is a recommendation for HR to review. It never rejects anyone.

Run:  python main.py
"""
import logging

from src.ai_extraction import analyse_stored_cvs, format_cv_info
from src.attachment_downloader import save_cv_attachments
from src.email_processor import fetch_new_application_emails, mark_processed
from src.gmail_client import GmailAuthError, get_gmail_service
from src.job_description import JobDescriptionError, load_job
from src.ollama_client import OllamaError, check_ollama
from src.ranking import format_report, save_report, screen_candidates
from src.text_extractor import build_cv_record, load_cv_records, save_cv_record

logger = logging.getLogger(__name__)


def process_email(service, email):
    """Download the CVs of one email, extract their text and store the records.

    The email is marked as processed only after at least one CV record has
    been stored. If nothing could be saved (for example a download error),
    it stays unmarked and is tried again on the next run.
    """
    records = []
    for download in save_cv_attachments(service, email):
        if download["status"] != "saved":
            logger.info("%s -> %s: %s", download["filename"],
                        download["status"], download["reason"])
            continue
        record = build_cv_record(download)
        record["record_file"] = save_cv_record(record)
        records.append(record)

    if records:
        mark_processed(email["message_id"])
    else:
        logger.warning("No CV stored for message %s; it will be retried next run",
                       email["message_id"])
    return records


def collect_cvs_from_gmail(max_results=50):
    """Run Steps 2-5 and return the stored CV records."""
    service = get_gmail_service()
    records = []
    for email in fetch_new_application_emails(service, max_results=max_results):
        records.extend(process_email(service, email))
    return records


def print_summary(records):
    if not records:
        print("\nNo new CVs to process.")
        return
    print(f"\nProcessed {len(records)} CV(s):")
    for record in records:
        candidate = record["candidate"]
        extraction = record["extraction"]
        print(f"\n--- {record['original_filename']} ---")
        print(f"Candidate : {candidate['name_from_email']} <{candidate['email_from_email']}>")
        print(f"Subject   : {record['email_subject']}")
        print(f"Status    : {extraction['status']} ({extraction['reason']})")
        print(f"Saved to  : {record['record_file']}")
        if extraction["status"] != "ok":
            print("            -> needs manual review by HR")


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    # Hide a harmless notice printed by Google's library.
    logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)
    # Hide the network and model-loading chatter of the AI libraries.
    for noisy in ("httpx", "sentence_transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        print_summary(collect_cvs_from_gmail())
    except GmailAuthError as e:
        print(f"\nGmail login problem: {e}")
        return

    # Step 7: structured information for every stored CV not analysed yet.
    try:
        check_ollama()
    except OllamaError as e:
        print(f"\nAI extraction skipped: {e}")
        return
    print("\nCandidate information (extracted by the local LLM):")
    for record in analyse_stored_cvs():
        print(f"\n=== {record['original_filename']} ===")
        if record.get("cv_info"):
            print(format_cv_info(record["cv_info"]))
        else:
            print(record.get("cv_info_error")
                  or "No readable text: needs manual review by HR")

    # Steps 8-9: compare every stored CV with the job, then group and rank.
    try:
        job = load_job()
    except JobDescriptionError as e:
        print(f"\nMatching skipped: {e}")
        return
    records = load_cv_records()
    if not records:
        print("\nNo stored CVs to compare with the job yet.")
        return
    report = screen_candidates(job, records)
    print("\n" + format_report(report))
    print(f"\nFull report with all evidence saved to: {save_report(report)}")


if __name__ == "__main__":
    main()
