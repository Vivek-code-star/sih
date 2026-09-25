"""
Mining/geological-report-specific extraction layer.
This is what makes answers domain-aware instead of generic text search —
e.g. pulling out drill hole IDs, assay grades, and coordinates so the QA
engine can quote exact numeric values instead of paraphrasing them
(paraphrased numbers are dangerous in a geological report).
"""
import regex as re
from typing import Dict, List

# Patterns tuned for common geological report phrasing.
PATTERNS = {
    "drill_hole_id": re.compile(r"\b(?:DH|BH|DDH)[-_]?\d{2,5}\b", re.IGNORECASE),
    "grade_gpt": re.compile(r"\b\d+(?:\.\d+)?\s*g/t\b", re.IGNORECASE),
    "grade_ppm": re.compile(r"\b\d+(?:\.\d+)?\s*ppm\b", re.IGNORECASE),
    "percent_grade": re.compile(r"\b\d+(?:\.\d+)?\s*%\s*(?:Cu|Fe|Au|Zn|Pb|Ni)\b", re.IGNORECASE),
    "coordinates": re.compile(r"\b\d{1,3}\.\d{3,6}[°]?\s*[NS]?,?\s*\d{1,3}\.\d{3,6}[°]?\s*[EW]?\b"),
    "depth": re.compile(r"\b\d+(?:\.\d+)?\s*(?:m|meters|metres|ft)\b", re.IGNORECASE),
    "standard_ref": re.compile(r"\b(NI\s?43-101|JORC(?:\s?2012)?)\b", re.IGNORECASE),
}


def extract_entities(text: str) -> Dict[str, List[str]]:
    found = {}
    for label, pattern in PATTERNS.items():
        matches = sorted(set(m.group(0) for m in pattern.finditer(text)))
        if matches:
            found[label] = matches
    return found


def keyword_hits(text: str, keywords: List[str]) -> List[str]:
    lower = text.lower()
    return [kw for kw in keywords if kw.lower() in lower]
