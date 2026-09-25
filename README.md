# Geo Mining AI Backend

## Python Environment
Use Python 3.12.x with a local `.venv`.

## Setup
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run
```powershell
uvicorn app.main:app --reload --port 8000
```

## Backend Flow
Upload
-> PDF extraction / page detection
-> OCR for scanned pages
-> chunking
-> embeddings
-> FAISS indexing
-> retrieval
-> extractive QA
-> geological entity cross-check
-> confidence
-> human review when required.

## Important
Tesseract OCR must be installed separately on Windows because pytesseract
uses the system Tesseract executable.
