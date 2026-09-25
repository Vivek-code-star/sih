"""
Step 4 of .pipeline: "Index" — chunks page text, embeds it, and stores it
in a FAISS index so /ask can retrieve the most relevant passages across
potentially many geological reports (multi-document search).
"""
import os
import pickle
import numpy as np
import faiss
from dataclasses import dataclass
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from .config import settings
from .pdf_processor import DocumentExtraction

_model: SentenceTransformer = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


@dataclass
class Chunk:
    doc_id: str
    filename: str
    page: int
    text: str


def _chunk_text(text: str, size: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(size - overlap, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start:start + size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if start + size >= len(words):
            break
    return chunks


def chunk_document(doc: DocumentExtraction) -> List[Chunk]:
    chunks = []
    for page in doc.pages:
        if not page.text:
            continue
        for piece in _chunk_text(page.text, settings.CHUNK_SIZE_TOKENS, settings.CHUNK_OVERLAP_TOKENS):
            chunks.append(Chunk(doc_id=doc.doc_id, filename=doc.filename, page=page.page_number, text=piece))
    return chunks


class VectorStore:
    """
    Thin wrapper around a single FAISS index shared across all uploaded
    reports, so a question can be answered from one document or searched
    across the whole corpus (doc_id=None case in AskRequest).
    """

    def __init__(self):
        self.index = None
        self.chunks: List[Chunk] = []
        self._index_path = os.path.join(settings.INDEX_DIR, "faiss.index")
        self._meta_path = os.path.join(settings.INDEX_DIR, "chunks.pkl")
        self._load_if_exists()

    def _load_if_exists(self):
        if os.path.exists(self._index_path) and os.path.exists(self._meta_path):
            self.index = faiss.read_index(self._index_path)
            with open(self._meta_path, "rb") as f:
                self.chunks = pickle.load(f)

    def _persist(self):
        faiss.write_index(self.index, self._index_path)
        with open(self._meta_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def add_document(self, doc: DocumentExtraction):
        new_chunks = chunk_document(doc)
        if not new_chunks:
            return
        model = get_model()
        vectors = model.encode([c.text for c in new_chunks], normalize_embeddings=True)
        vectors = np.array(vectors, dtype="float32")

        if self.index is None:
            dim = vectors.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized inner product

        self.index.add(vectors)
        self.chunks.extend(new_chunks)
        self._persist()

    def search(self, query: str, top_k: int, doc_id: str = None) -> List[Tuple[Chunk, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        model = get_model()
        q_vec = np.array(model.encode([query], normalize_embeddings=True), dtype="float32")

        # Over-fetch then filter by doc_id, since FAISS flat index has no metadata filter built in.
        fetch_k = top_k * 5 if doc_id else top_k
        scores, idxs = self.index.search(q_vec, min(fetch_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            if doc_id and chunk.doc_id != doc_id:
                continue
            results.append((chunk, float(score)))
            if len(results) >= top_k:
                break
        return results


vector_store = VectorStore()
