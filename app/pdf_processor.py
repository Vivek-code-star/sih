"""
Step 1-2 of the pipeline shown in .pipeline / .step:
  "Upload"  ->  "Detect page type"  ->  hands off scanned pages to ocr_processor.py

A geological report PDF is often MIXED: some pages are native text (e.g. cover
page, table of contents, written sections) and some are scanned drill logs,
maps, or old assay certificates. So detection happens PER PAGE, not per file.
"""
import fitz  # PyMuPDF
from dataclasses import dataclass
from typing import List
from .config import settings


@dataclass
class PageContent:
    page_number: int          # 1-indexed, for citing sources back to the UI
    text: str
    is_scanned: bool
    image_bytes: bytes = None  # populated only if is_scanned, for OCR step


@dataclass
class DocumentExtraction:
    doc_id: str
    filename: str
    pages: List[PageContent]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def scanned_page_count(self) -> int:
        return sum(1 for p in self.pages if p.is_scanned)


def extract_document(doc_id: str, filename: str, file_path: str) -> DocumentExtraction:
    """
    Opens the PDF, pulls native text per page, and flags pages that are
    image-only (no extractable text layer) so they can be routed to OCR.
    """
    pdf = fitz.open(file_path)
    pages: List[PageContent] = []

    for i, page in enumerate(pdf, start=1):
        native_text = page.get_text("text").strip()

        if len(native_text) >= settings.MIN_TEXT_CHARS_PER_PAGE:
            # Normal PDF page - text layer is usable directly.
            pages.append(PageContent(page_number=i, text=native_text, is_scanned=False))
        else:
            # Scanned / image-based page - no usable text layer.
            # Render at high DPI so OCR accuracy holds up on small map labels
            # and assay-table numbers, which are common failure points.
            zoom = settings.OCR_DPI / 72
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)
            image_bytes = pix.tobytes("png")
            pages.append(PageContent(
                page_number=i, text="", is_scanned=True, image_bytes=image_bytes
            ))

    pdf.close()
    return DocumentExtraction(doc_id=doc_id, filename=filename, pages=pages)
