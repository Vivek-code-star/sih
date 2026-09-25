"""
Central configuration for the AIML backend.
Values here are surfaced/edited via the frontend Settings page (.settings panel).
"""
import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # --- storage ---
    UPLOAD_DIR: str = "./storage/uploads"
    INDEX_DIR: str = "./storage/index"

    # --- OCR / scan detection ---
    # If extracted native text per page is below this many characters,
    # the page is treated as "scanned" and routed to OCR.
    MIN_TEXT_CHARS_PER_PAGE: int = 40
    OCR_DPI: int = 300
    OCR_LANG: str = "eng"

    # --- chunking ---
    CHUNK_SIZE_TOKENS: int = 350
    CHUNK_OVERLAP_TOKENS: int = 60

    # --- embeddings / retrieval ---
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    TOP_K_RETRIEVAL: int = 6

    # --- answer generation (fully local AIML model, no external API) ---
    # Extractive QA model: reads the retrieved passage and points to the
    # exact span containing the answer (works well for factual numbers
    # like grades, depths, hole IDs — the common questions on geo reports).
    QA_MODEL_NAME: str = "deepset/roberta-base-squad2"
    QA_MAX_ANSWER_LEN: int = 60
    QA_DEVICE: int = -1  # -1 = CPU, 0 = first GPU (set to 0 if CUDA available)

    # How many top retrieved chunks to actually run the QA model over.
    # Running it over more chunks = slower but more likely to find the
    # right answer if it's buried in a lower-ranked passage.
    QA_CANDIDATE_CHUNKS: int = 4

    # --- confidence / audit routing (drives .audit-flow and .review cards) ---
    # Below this score, an answer is auto-flagged into the human review queue
    # instead of being shown as final.
    CONFIDENCE_AUTO_ACCEPT: float = 0.80
    CONFIDENCE_REVIEW_THRESHOLD: float = 0.55  # below this -> "warning" badge in UI

    # --- geological/mining domain vocabulary (used by domain_extractor.py) ---
    DOMAIN_KEYWORDS = [
        "assay", "drill hole", "borehole", "ore grade", "cut-off grade",
        "lithology", "strike", "dip", "tonnage", "ppm", "g/t", "au", "cu", "fe",
        "overburden", "seam", "stratigraphy", "mineral resource", "mineral reserve",
        "NI 43-101", "JORC", "core sample", "geochemical", "outcrop",
        "waste rock", "stripping ratio", "beneficiation", "concentrate",
    ]


settings = Settings()
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.INDEX_DIR, exist_ok=True)
