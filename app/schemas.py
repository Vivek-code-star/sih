"""
Schemas = the exact JSON contract your frontend JS should expect.
Every field here maps to something rendered in the CSS you shared.
"""
from typing import List, Optional, Literal
from pydantic import BaseModel


# ---------- Upload / Pipeline (.pipeline, .step) ----------

class PipelineStepStatus(BaseModel):
    step: Literal["upload", "detect", "ocr_extract", "index", "ready"]
    label: str          # shown as <b> in .step
    detail: str         # shown as <span> in .step
    done: bool


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    page_count: int
    scanned_page_count: int      # pages that needed OCR
    pipeline: List[PipelineStepStatus]


# ---------- Dashboard cards (.cards / .card) ----------

class MetricCard(BaseModel):
    icon: str
    label: str
    value: str
    sub: str


class DashboardMetrics(BaseModel):
    cards: List[MetricCard]


# ---------- Q&A / Insight panel (.panel, .insight, .confidence, .progress) ----------

class AskRequest(BaseModel):
    doc_id: Optional[str] = None   # None = search across all indexed reports
    question: str


class SourceChunk(BaseModel):
    doc_id: str
    filename: str
    page: int
    snippet: str
    similarity: float


class AskResponse(BaseModel):
    question: str
    answer: str
    confidence: float                 # 0.0 - 1.0, drives .progress width
    status: Literal["success", "warning"]   # maps to .success / .warning badges
    needs_review: bool                 # true -> pushed into audit queue
    sources: List[SourceChunk]
    extracted_entities: dict           # geological fields pulled via regex/NER


# ---------- Audit / human-in-the-loop (.audit-flow, .review) ----------

class ReviewItem(BaseModel):
    review_id: str
    question: str
    proposed_answer: str
    confidence: float
    doc_id: str
    filename: str


class ReviewDecision(BaseModel):
    review_id: str
    decision: Literal["approve", "reject", "edit"]
    corrected_answer: Optional[str] = None


# ---------- Settings (.settings) ----------

class SettingsPayload(BaseModel):
    ocr_lang: str
    confidence_auto_accept: float
    confidence_review_threshold: float
    top_k_retrieval: int
    answer_model: str
