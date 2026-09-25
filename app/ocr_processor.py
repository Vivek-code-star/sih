"""
Step 3 of .pipeline: "OCR Extract" — runs only on pages pdf_processor.py
flagged as scanned. Geological reports scanned from old field notes/maps
often have low contrast and rotated text, so we do light preprocessing
before OCR to keep accuracy usable for numeric assay values.
"""
import io
import pytesseract
from PIL import Image, ImageOps, ImageFilter
from .config import settings
from .pdf_processor import DocumentExtraction


def _preprocess(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    gray = gray.filter(ImageFilter.SHARPEN)
    return gray


def ocr_page(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes))
    image = _preprocess(image)
    text = pytesseract.image_to_string(image, lang=settings.OCR_LANG, config="--psm 6")
    return text.strip()


def run_ocr_on_document(doc: DocumentExtraction) -> DocumentExtraction:
    """
    Fills in `.text` for every page that was flagged is_scanned=True.
    Mutates and returns the same DocumentExtraction so the pipeline
    can report per-page progress back to the frontend if needed.
    """
    for page in doc.pages:
        if page.is_scanned and page.image_bytes:
            page.text = ocr_page(page.image_bytes)
    return doc
