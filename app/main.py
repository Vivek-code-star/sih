"""
FastAPI entrypoint. Every route here corresponds to a UI element in the
CSS you provided. Run with:
    uvicorn app.main:app --reload --port 8000

Frontend JS just needs to call these endpoints — e.g.:
    fetch('/api/ask', {method:'POST', body: JSON.stringify({question})})
"""
import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .schemas import (
    AskRequest, AskResponse, DashboardMetrics, MetricCard,
    ReviewDecision, SettingsPayload, UploadResponse,
)
from . import pipeline, qa_engine, audit
from .embeddings import vector_store

app = FastAPI(title="Geo/Mining Report QA — AIML Backend")

# Allow the dashboard frontend (served separately) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your actual frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Upload / Pipeline (.upload-btn, .pipeline) ----------

@app.post("/api/upload", response_model=UploadResponse)
async def upload_report(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    dest_path = os.path.join(settings.UPLOAD_DIR, f"{doc_id}.pdf")
    with open(dest_path, "wb") as f:
        f.write(await file.read())

    return pipeline.process_upload(doc_id, file.filename, dest_path)


# ---------- Ask a question (.big-search, .panel, .insight, .confidence) ----------

@app.post("/api/ask", response_model=AskResponse)
async def ask(payload: AskRequest):
    if not payload.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    result = qa_engine.answer_question(payload.question, doc_id=payload.doc_id)
    return AskResponse(
        question=payload.question,
        answer=result.answer,
        confidence=result.confidence,
        status=result.status,
        needs_review=result.needs_review,
        sources=result.sources,
        extracted_entities=result.entities,
    )


# ---------- Dashboard metric cards (.cards, .card) ----------

@app.get("/api/metrics", response_model=DashboardMetrics)
async def metrics():
    total_chunks = len(vector_store.chunks)
    unique_docs = len({c.doc_id for c in vector_store.chunks})
    pending_reviews = len(audit.list_pending())

    cards = [
        MetricCard(icon="📄", label="Reports Indexed", value=str(unique_docs), sub="Processed"),
        MetricCard(icon="🔎", label="Searchable Chunks", value=str(total_chunks), sub="Indexed passages"),
        MetricCard(icon="🕵️", label="Pending Reviews", value=str(pending_reviews), sub="Needs human check"),
        MetricCard(icon="🎯", label="Auto-Accept Threshold", value=f"{settings.CONFIDENCE_AUTO_ACCEPT:.0%}", sub="Confidence cutoff"),
    ]
    return DashboardMetrics(cards=cards)


# ---------- Audit / review queue (.audit-flow, .review) ----------

@app.get("/api/audit/queue")
async def review_queue():
    return audit.list_pending()


@app.post("/api/audit/decide")
async def review_decide(decision: ReviewDecision):
    ok = audit.resolve(decision.review_id, decision.decision, decision.corrected_answer)
    if not ok:
        raise HTTPException(404, "Review item not found.")
    return {"status": "ok"}


# ---------- Settings (.settings) ----------

@app.get("/api/settings", response_model=SettingsPayload)
async def get_settings():
    return SettingsPayload(
        ocr_lang=settings.OCR_LANG,
        confidence_auto_accept=settings.CONFIDENCE_AUTO_ACCEPT,
        confidence_review_threshold=settings.CONFIDENCE_REVIEW_THRESHOLD,
        top_k_retrieval=settings.TOP_K_RETRIEVAL,
        answer_model=settings.QA_MODEL_NAME,
    )


@app.post("/api/settings")
async def update_settings(payload: SettingsPayload):
    settings.OCR_LANG = payload.ocr_lang
    settings.CONFIDENCE_AUTO_ACCEPT = payload.confidence_auto_accept
    settings.CONFIDENCE_REVIEW_THRESHOLD = payload.confidence_review_threshold
    settings.TOP_K_RETRIEVAL = payload.top_k_retrieval
    settings.QA_MODEL_NAME = payload.answer_model
    return {"status": "ok"}


@app.get("/api/health")
async def health():
    return {"status": "ok"}
