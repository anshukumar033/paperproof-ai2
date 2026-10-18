# PaperProof AI — Hackathon MVP

A Flask web app that compares submitted text with `.txt` documents in `reference_documents/`, surfaces candidate passages, and generates a plain-text report.

## Requirements
- Python 3.10+
- Internet access for initial dependency installation

## Setup

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

### Windows PowerShell
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

## Use
- Paste at least 20 characters or upload a `.txt` file (maximum 2 MB).
- Add additional plain-text reference papers to `reference_documents/`.
- Analyze and download the report.

## Scope and limitations
- Only local reference `.txt` files are compared; this does not search the internet.
- Similarity uses TF-IDF/cosine similarity and a prototype threshold; the percentage shown is the share of analyzed sentences flagged, **not** a universal plagiarism score.
- Citation detection is a simple heuristic and does not verify citation correctness.
- Results are candidate signals for human review, not a plagiarism verdict.
- PDF/DOCX extraction, semantic embeddings, and live scholarly source search are not implemented in this MVP.

## Demo tips
1. Submit a sentence copied from `reference_documents/reference_1.txt`.
2. Submit unrelated text and compare results.
3. Explain the limitations honestly.
