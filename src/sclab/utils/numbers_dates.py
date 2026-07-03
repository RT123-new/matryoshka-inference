from __future__ import annotations

import re

NUMBER_RE = re.compile(r"(?<!\w)(?:[$]?\d+(?:,\d{3})*(?:\.\d+)?%?|GBP\s*\d+(?:,\d{3})*(?:\.\d+)?)(?!\w)")
DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|"
    r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b",
    re.IGNORECASE,
)
ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,4}\b")


def extract_numbers(text: str) -> list[str]:
    return list(dict.fromkeys(match.group(0) for match in NUMBER_RE.finditer(text)))


def extract_dates(text: str) -> list[str]:
    return list(dict.fromkeys(match.group(0) for match in DATE_RE.finditer(text)))


def extract_entities(text: str) -> list[str]:
    stop = {"The", "A", "An", "If", "Source", "Question", "Answer", "Rules"}
    entities = []
    for match in ENTITY_RE.finditer(text):
        value = match.group(0).strip()
        if value in stop or len(value) < 3:
            continue
        entities.append(value)
    return list(dict.fromkeys(entities))


def normalize_number(value: str) -> str:
    return re.sub(r"[,$\s%]", "", value.lower().replace("gbp", ""))
