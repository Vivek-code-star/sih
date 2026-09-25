"""
Orchestrates the full upload -> answer-ready flow and reports status in the
exact shape .pipeline / .step expects, so the frontend can light up each
step as it completes.
"""
from .pdf_processor import extract_document
from .ocr_processor import run_ocr_on_document
from .embeddings import vector_store
from .schemas import PipelineStepStatus, UploadResponse


def process_upload(doc_id: str, filename: str, file_path: str) -> UploadResponse:
    steps = []

    # Step 1: Upload (already done by the time this runs)
    steps.append(PipelineStepStatus(step="upload", label="Upload", detail=filename, done=True))

    # Step 2: Detect normal vs scanned pages
    doc = extract_document(doc_id, filename, file_path)
    steps.append(PipelineStepStatus(
        step="detect", label="Detect Page Type",
        detail=f"{doc.scanned_page_count} of {doc.page_count} pages scanned",
        done=True,
    ))

    # Step 3: OCR the scanned pages only
    if doc.scanned_page_count > 0:
        doc = run_ocr_on_document(doc)
        steps.append(PipelineStepStatus(
            step="ocr_extract", label="OCR Extract",
            detail=f"{doc.scanned_page_count} pages OCR'd", done=True,
        ))
    else:
        steps.append(PipelineStepStatus(
            step="ocr_extract", label="OCR Extract",
            detail="Skipped — all pages had native text", done=True,
        ))

    # Step 4: Chunk + embed + index
    vector_store.add_document(doc)
    steps.append(PipelineStepStatus(
        step="index", label="Index", detail="Embedded and added to search index", done=True,
    ))

    # Step 5: Ready
    steps.append(PipelineStepStatus(
        step="ready", label="Ready", detail="Available for Q&A", done=True,
    ))

    return UploadResponse(
        doc_id=doc_id,
        filename=filename,
        page_count=doc.page_count,
        scanned_page_count=doc.scanned_page_count,
        pipeline=steps,
    )
