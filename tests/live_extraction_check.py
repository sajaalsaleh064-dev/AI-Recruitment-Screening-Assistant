"""Step 7 live check: run the REAL model on the sample CVs and job descriptions.

Unlike the unit tests, this needs Ollama running with the model downloaded,
and takes a minute or two. Use it to see how good the prompts are.

Run:  python -m tests.live_extraction_check
"""
import logging
import os
import time

from src import ai_extraction as ai
from src.config import JOB_DESCRIPTION_PATH, OLLAMA_MODEL
from src.job_description import format_job
from src.ollama_client import OllamaError, check_ollama

SAMPLES = os.path.join(os.path.dirname(__file__), "sample_data")
CVS = ["cv_strong.txt", "cv_partial.txt", "cv_weak.txt"]
JOBS = [JOB_DESCRIPTION_PATH, os.path.join(SAMPLES, "job_data_analyst.txt")]


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
    try:
        check_ollama()
    except OllamaError as e:
        print(f"[FAIL] {e}")
        return
    print(f"Model: {OLLAMA_MODEL}")

    for path in JOBS:
        started = time.time()
        info = ai.extract_job_requirements(read(path))
        print(f"\n===== JOB: {path} ({time.time() - started:.0f}s) =====")
        print(format_job(ai.job_from_extraction(info)))
        if info["unverified"]:
            print("Unverified (not used):", info["unverified"])

    for name in CVS:
        started = time.time()
        info = ai.extract_cv_information(read(os.path.join(SAMPLES, name)))
        print(f"\n===== CV: {name} ({time.time() - started:.0f}s) =====")
        print(ai.format_cv_info(info))


if __name__ == "__main__":
    main()
