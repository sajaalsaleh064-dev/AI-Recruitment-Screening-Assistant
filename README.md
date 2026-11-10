# AI Recruitment Screening Assistant

A Python tool that helps HR teams screen job applications faster. It reads
application emails from Gmail, extracts the text of attached CVs, uses a local
LLM to structure the candidate's information, compares each candidate with the
job requirements, and produces a ranked report with evidence.

**The tool assists HR. It never makes a hiring decision and never rejects
anyone automatically.** Every result comes with the lines of the CV that
support it, so a person can check and overrule it.

## Workflow

```
Gmail (read-only)
  -> application emails          src/email_processor.py
  -> CV attachments (PDF/DOCX)   src/attachment_downloader.py
  -> clean CV text               src/text_extractor.py
  -> structured candidate info   src/ai_extraction.py  (Ollama, JSON output)
  -> requirement-by-requirement  src/matching.py       (rules + embeddings)
  -> Must-Have filter + ranking  src/ranking.py
  -> report for HR               data/output/screening_report.json
```

## Technologies

| Purpose | Technology |
|---|---|
| Email access | Gmail API, OAuth 2.0 (read-only scope), google-api-python-client |
| CV text | pypdf (PDF), python-docx (DOCX) |
| Information extraction | Ollama with a local LLM (`llama3.1`), structured JSON output |
| Semantic matching | Sentence Transformers `all-MiniLM-L6-v2`, cosine similarity |
| Configuration | `.env` file (python-dotenv) |

## Project structure

```
main.py                     runs the whole workflow
requirements.txt
.env.example                template for your .env
job/
  job_description.txt       the job advert (free text)
  job_requirements.json     structured requirements, Must-Have or optional
src/
  config.py                 loads settings from .env
  gmail_client.py           Gmail login (OAuth 2.0, read-only)
  email_processor.py        search, parse, duplicate tracking
  attachment_downloader.py  download and validate PDF / DOCX files
  text_extractor.py         PDF / DOCX -> clean text + candidate record
  job_description.py        load, validate and review job requirements
  ollama_client.py          connection to the local LLM
  ai_extraction.py          prompts, JSON validation, grounding check
  matching.py               embeddings, cosine similarity, matching rules
  ranking.py                Must-Have filtering, scoring, ranking, report
tests/
  test_*.py                 automated tests (run offline)
  live_extraction_check.py  runs the real LLM on the sample data
  sample_candidates.py      three fictional candidates for the tests
  sample_data/              sample CVs and a second job description
data/                       downloaded CVs and results (ignored by git)
```

## Installation

Requirements: Python 3.10+, [Ollama](https://ollama.com), a Google account.

```powershell
git clone <your repository URL>
cd ai-recruitment-assistant
python -m venv venv
.\venv\Scripts\Activate.ps1          # macOS / Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env               # macOS / Linux: cp .env.example .env
ollama pull llama3.1
python -m tests.test_setup
```

The last command should print `[ok]` for every library and `All good!`.

## Gmail setup (once)

1. Open <https://console.cloud.google.com> and create a project.
2. **APIs & Services -> Library** -> search "Gmail API" -> **Enable**.
3. **APIs & Services -> OAuth consent screen**: user type **External**, then
   add your own Gmail address under **Test users**.
4. **Credentials -> Create credentials -> OAuth client ID** -> application
   type **Desktop app**.
5. Download the JSON file, rename it to exactly `credentials.json` and put it
   in the project folder (next to `requirements.txt`).
6. Run `python -m src.gmail_client`. A browser window opens; approve the
   permission "View your email messages and settings".

Expected output:

```
[ok] Connected to: your.address@gmail.com
[ok] Total messages: 36
```

A `token.json` file is created so you do not log in again. The app requests
only the `gmail.readonly` scope: it cannot send, delete or modify email.

## Usage

```powershell
python main.py
```

This finds new application emails, downloads and reads the CVs, extracts the
candidate information, compares it with the job and prints the report.

Each step can also be run on its own:

| Command | What it does |
|---|---|
| `python -m src.gmail_client` | Test the Gmail connection |
| `python -m src.email_processor` | List new application emails |
| `python -m src.attachment_downloader` | Download their CV attachments |
| `python -m src.text_extractor <file>` | Show the cleaned text of one CV |
| `python -m src.job_description` | Show the job requirements |
| `python -m src.job_description --review` | Let HR choose the Must-Haves |
| `python -m src.ollama_client` | Test the Ollama connection |
| `python -m src.ai_extraction job` | Extract requirements from the job text |
| `python -m src.ai_extraction cvs` | Extract information from stored CVs |
| `python -m src.matching` | Requirement-by-requirement results |
| `python -m src.ranking` | Must-Have filtering and ranking report |

An email counts as an application when it has a PDF or DOCX attachment and
mentions an application-related word. The Gmail search itself is set by
`GMAIL_QUERY` in `.env`. Processed message IDs are stored in
`data/processed_ids.json`, so an email is never processed twice; delete that
file to process everything again.

## How candidates are evaluated

For every requirement a candidate gets one of:

- **Meets the Requirement**
- **Partially Meets the Requirement**
- **Does Not Meet the Requirement**

each with an explanation and evidence quoted from the CV. If nothing in the CV
supports a requirement the evidence reads **No evidence found**.

Rules and semantic similarity are combined:

| Category | Rule | Semantic similarity |
|---|---|---|
| Skill | Skill from the CV is named in the requirement | Otherwise the closest skill or sentence by meaning |
| Experience | Years of related jobs vs. the minimum | Decides which jobs are related |
| Education | Degree level (diploma < bachelor < master < PhD) | Decides whether the field matches |
| Language | Language listed, level compared | - |
| Certification | Certificate named in the requirement | Otherwise the closest certificate |

Candidates are then placed in three groups:

1. **Meets all Must-Have requirements** - ranked by score, best first.
2. **Needs further review** - a Must-Have is only partially met, or the CV
   could not be read automatically.
3. **Does not meet one or more Must-Have requirements** - kept in the report
   with the reasons; not rejected automatically.

Score (0-100): Meets = 1 point, Partially = 0.5, Does Not Meet = 0, and a
Must-Have weighs twice an optional requirement.

## Protection against invented information

- The LLM only extracts; scoring is done by rules and embeddings.
- The prompt forbids guessing, and a JSON schema fixes the answer's structure.
- Every extracted item must appear in the CV text. Items that do not are moved
  to an `unverified` list and are not used.
- A requirement is never counted as met without a supporting line from the CV.
- Wish statements such as "interested in learning Python" are not evidence.

## Tests

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

155 tests covering email processing, attachment download (PDF and DOCX), text
extraction, job description handling, JSON validation, AI extraction, rule
based matching, embeddings, cosine similarity, Must-Have filtering and
ranking. They use a fake Gmail service and a fake LLM, so they need neither
credentials nor Ollama. The matching tests load the real embedding model
(downloaded once, about 90 MB).

To check the real LLM on the sample CVs and job descriptions:

```powershell
python -m tests.live_extraction_check
```

Sample candidates (`tests/sample_data/`), against the Python Developer job:

| Candidate | Met | Partial | Not met | Group |
|---|---|---|---|---|
| Layla Haddad (strong) | 12 | 0 | 0 | Meets all Must-Haves, score 100 |
| Omar Khalil (partial) | 4 | 6 | 2 | Needs further review |
| Rana Mansour (weak) | 1 | 0 | 11 | Does not meet Must-Haves |

All names, companies and CVs in this repository are fictional test data.

## Security and privacy

- `credentials.json`, `token.json`, `.env` and everything under `data/` are
  listed in `.gitignore`. Never commit them.
- Gmail access is read-only.
- CVs are processed locally: the LLM (Ollama) and the embedding model both run
  on your own computer, so candidate data is not sent to an AI service.
- Attachment names from emails are sanitised, and file contents are checked
  before saving (real PDF header / real DOCX archive, 10 MB limit).

## Known limitations

- Scanned PDFs have no text layer and are flagged for manual review (no OCR).
- Old `.doc` files are not supported, only `.docx`.
- Similarity values between 0.35 and 0.40 are noisy and can produce a loose
  "Partially Meets"; the quoted evidence makes these easy to spot.
- The similarity thresholds were tuned on the sample CVs. Adjust them in
  `.env` (`SIMILARITY_MEETS`, `SIMILARITY_PARTIAL`, `SIMILARITY_RELEVANT`).
- The LLM takes 30-60 seconds per document on a laptop.

## Troubleshooting

| Problem | Fix |
|---|---|
| `'credentials.json' not found` | Wrong name or folder. Check it is not `credentials.json.json`. |
| `Error 403: access_denied` at login | Add your address under OAuth consent screen -> Test users. |
| `invalid_grant` after about a week | Delete `token.json` and run again (test-mode tokens expire). |
| `No new application emails found` | Loosen `GMAIL_QUERY` in `.env`, or delete `data/processed_ids.json`. |
| `Cannot reach Ollama` | Start the Ollama app or run `ollama serve`. |
| `Model 'llama3.1' is not downloaded` | Run `ollama pull llama3.1`. |
| `running scripts is disabled` (PowerShell) | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
