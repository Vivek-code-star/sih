"""
Backs the .audit-flow / .review UI: a simple in-memory queue of
low-confidence answers awaiting human approve/reject/edit.
Swap the in-memory dict for a real DB table in production.
"""
import uuid
from typing import List, Optional
from .schemas import ReviewItem

_queue: dict[str, ReviewItem] = {}


def enqueue_review(question: str, proposed_answer: str, confidence: float,
                    doc_id: str, filename: str) -> ReviewItem:
    review_id = str(uuid.uuid4())
    item = ReviewItem(
        review_id=review_id,
        question=question,
        proposed_answer=proposed_answer,
        confidence=confidence,
        doc_id=doc_id,
        filename=filename,
    )
    _queue[review_id] = item
    return item


def list_pending() -> List[ReviewItem]:
    return list(_queue.values())


def resolve(review_id: str, decision: str, corrected_answer: Optional[str] = None) -> bool:
    if review_id not in _queue:
        return False
    if decision in ("approve", "edit"):
        # In production: write the (possibly corrected) answer back into a
        # verified-answers store so future identical questions skip the
        # LLM call entirely and return the human-verified answer.
        pass
    del _queue[review_id]
    return True
