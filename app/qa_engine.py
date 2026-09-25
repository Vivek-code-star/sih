"""
CORE AIML LOGIC — fully local, no external API.

Flow to predict the correct answer to a question, from ANY uploaded
mining/geological report (normal-text OR OCR'd scanned pages, since both
were indexed identically by embeddings.py):

  1. RETRIEVAL  -> semantic search (FAISS + sentence-transformers) finds
                   the passages most likely to contain the answer.
  2. EXTRACTIVE QA -> a transformer model (RoBERTa fine-tuned on SQuAD2)
                   reads each candidate passage and points to the exact
                   answer span inside it, with its own confidence score.
  3. ENTITY CROSS-CHECK -> domain_extractor.py verifies the picked answer
                   span against regex-extracted geological entities
                   (grades, hole IDs, coordinates) when the question is
                   asking for that kind of value.
  4. CONFIDENCE FUSION -> combines retrieval similarity + QA model score
                   + entity-grounding into one 0-1 confidence used to
                   drive the .confidence / .progress bar and the
                   .audit-flow review queue.
"""
from dataclasses import dataclass
from typing import List, Dict
from transformers import pipeline as hf_pipeline

from .config import settings
from .embeddings import vector_store
from .domain_extractor import extract_entities
from . import audit

_qa_pipeline = None


def get_qa_pipeline():
    """
    Lazy-loads the local extractive QA model once per process.
    This IS the "AIML model" — a transformer trained for question
    answering (start/end span prediction over a context passage).
    """
    global _qa_pipeline
    if _qa_pipeline is None:
        _qa_pipeline = hf_pipeline(
            "question-answering",
            model=settings.QA_MODEL_NAME,
            tokenizer=settings.QA_MODEL_NAME,
            device=settings.QA_DEVICE,
        )
    return _qa_pipeline


@dataclass
class AnswerResult:
    answer: str
    confidence: float
    status: str            # "success" | "warning"
    needs_review: bool
    sources: List[dict]
    entities: Dict[str, List[str]]


def _looks_numeric_question(question: str) -> bool:
    q = question.lower()
    triggers = ["grade", "depth", "coordinate", "ppm", "g/t", "how much",
                "how many", "hole id", "borehole", "assay", "percent", "%"]
    return any(t in q for t in triggers)


def _score_confidence(retrieval_sim: float, qa_score: float,
                       answer_text: str, entities: Dict, numeric_q: bool) -> float:
    """
    Fuses three independent signals into one confidence score:
      - retrieval_sim : did we even find the right passage? (0-1, cosine sim)
      - qa_score       : the transformer's own softmax confidence in the
                          answer span it picked (0-1)
      - grounding      : for numeric/factual questions, does the picked
                          answer actually match a regex-extracted entity
                          from the source text? Guards against the model
                          confidently pointing at the wrong number.
    """
    retrieval_component = max(0.0, min(retrieval_sim, 1.0))
    qa_component = max(0.0, min(qa_score, 1.0))

    if numeric_q:
        all_entities = [v for values in entities.values() for v in values]
        grounded = any(e.lower() in answer_text.lower() or answer_text.lower() in e.lower()
                       for e in all_entities)
        grounding_component = 1.0 if grounded else 0.4
    else:
        grounding_component = 1.0  # not applicable for descriptive questions

    confidence = 0.4 * retrieval_component + 0.4 * qa_component + 0.2 * grounding_component
    return round(min(max(confidence, 0.0), 1.0), 3)


def answer_question(question: str, doc_id: str = None) -> AnswerResult:
    # 1. RETRIEVAL — works identically whether the source page was native
    #    text or OCR'd, because both were embedded the same way at index time.
    chunks_with_scores = vector_store.search(question, settings.QA_CANDIDATE_CHUNKS, doc_id=doc_id)

    if not chunks_with_scores:
        return AnswerResult(
            answer="No indexed report content matches this question yet. Upload and process a report first.",
            confidence=0.0,
            status="warning",
            needs_review=True,
            sources=[],
            entities={},
        )

    qa_model = get_qa_pipeline()
    numeric_q = _looks_numeric_question(question)

    # 2. EXTRACTIVE QA — run the model over each candidate passage and
    #    keep the single best-scoring answer span across all of them.
    best = None  # (answer_text, qa_score, chunk, retrieval_sim)
    for chunk, retrieval_sim in chunks_with_scores:
        try:
            result = qa_model(question=question, context=chunk.text,
                               max_answer_len=settings.QA_MAX_ANSWER_LEN)
        except Exception:
            continue
        if best is None or result["score"] > best[1]:
            best = (result["answer"], result["score"], chunk, retrieval_sim)

    if best is None:
        return AnswerResult(
            answer="The QA model could not extract a confident answer from the retrieved passages.",
            confidence=0.0,
            status="warning",
            needs_review=True,
            sources=[{"doc_id": c.doc_id, "filename": c.filename, "page": c.page,
                      "snippet": c.text[:280], "similarity": round(s, 3)}
                     for c, s in chunks_with_scores],
            entities={},
        )

    answer_text, qa_score, best_chunk, retrieval_sim = best

    # 3. ENTITY CROSS-CHECK on the winning passage
    entities = extract_entities(best_chunk.text)

    # 4. CONFIDENCE FUSION
    confidence = _score_confidence(retrieval_sim, qa_score, answer_text, entities, numeric_q)
    status = "success" if confidence >= settings.CONFIDENCE_REVIEW_THRESHOLD else "warning"
    needs_review = confidence < settings.CONFIDENCE_AUTO_ACCEPT

    sources = [{
        "doc_id": c.doc_id,
        "filename": c.filename,
        "page": c.page,
        "snippet": c.text[:280],
        "similarity": round(s, 3),
    } for c, s in chunks_with_scores]

    if needs_review:
        audit.enqueue_review(
            question=question,
            proposed_answer=answer_text,
            confidence=confidence,
            doc_id=best_chunk.doc_id,
            filename=best_chunk.filename,
        )

    return AnswerResult(
        answer=answer_text,
        confidence=confidence,
        status=status,
        needs_review=needs_review,
        sources=sources,
        entities=entities,
    )
