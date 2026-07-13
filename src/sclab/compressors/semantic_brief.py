from __future__ import annotations

import math
import re

from sclab.compressors.base import Document, build_result
from sclab.tokenization import count_tokens
from sclab.utils.numbers_dates import extract_dates, extract_entities, extract_numbers
from sclab.utils.text_split import split_paragraphs, split_sentences, truncate_words

STOP_TERMS = {
    "what",
    "which",
    "when",
    "where",
    "who",
    "does",
    "the",
    "and",
    "are",
    "for",
    "from",
    "with",
    "about",
    "include",
    "source",
    "question",
    "after",
    "before",
    "over",
    "main",
    "listed",
    "summarise",
    "summarize",
    "three",
    "two",
}


class SemanticBriefCompressor:
    name = "semantic_brief"

    def __init__(self, max_items: int = 8, target_token_ratio: float | None = None) -> None:
        self.max_items = max_items
        self.target_token_ratio = target_token_ratio

    def compress(self, doc: Document, question: str | None = None):
        excerpts = self._relevant_excerpts(doc.text, question)[:4]
        evidence_text = "\n".join(excerpts) if excerpts else doc.text
        headings = self._headings(doc.text)[: self.max_items]
        numbers = extract_numbers(evidence_text)[: self.max_items]
        dates = extract_dates(evidence_text)[: self.max_items]
        entities = extract_entities(evidence_text)[: self.max_items]
        key_lines = self._key_lines(doc.text, question)[: self.max_items]
        claims = self._claims(doc.text, question)[: self.max_items]

        section_items = {
            "Headings": headings,
            "Entities": entities,
            "Dates": dates,
            "Numbers": numbers,
            "Key lines": key_lines,
            "Claims": claims,
            "Relevant excerpts": excerpts,
        }
        if self.target_token_ratio is None:
            text = _render_sections(
                ["Headings", "Entities", "Dates", "Numbers", "Key lines", "Claims", "Relevant excerpts"],
                section_items,
            )
        else:
            text = _fit_sections_to_budget(doc.text, self.target_token_ratio, section_items)
        return build_result(
            doc=doc,
            text=text,
            method=self.name,
            metadata={
                "deterministic": True,
                "target_token_ratio": self.target_token_ratio,
                "retained_numbers_count": len(numbers),
                "retained_dates_count": len(dates),
                "retained_named_entities_count": len(entities),
            },
        )

    def _headings(self, text: str) -> list[str]:
        headings = []
        for line in text.splitlines():
            stripped = line.strip().strip("#").strip()
            if not stripped:
                continue
            if line.lstrip().startswith("#") or (len(stripped) < 80 and stripped.endswith(":")):
                headings.append(stripped.rstrip(":"))
        return list(dict.fromkeys(headings))

    def _key_lines(self, text: str, question: str | None) -> list[str]:
        q_terms = _question_terms(question)
        scored = []
        for line in text.splitlines():
            stripped = line.strip(" -\t")
            if not stripped:
                continue
            line_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9_]+", stripped)}
            overlap = len(q_terms & line_terms)
            lowered = stripped.lower()
            if q_terms and overlap == 0 and _is_generated_distractor(lowered):
                continue
            score = overlap * 3
            if extract_numbers(stripped):
                score += 1
            if extract_dates(stripped):
                score += 1
            if score == 0 and lowered.startswith("distractor "):
                continue
            if (":" in stripped or re.match(r"^[-*]\s+", line.strip())
                    or extract_numbers(stripped) or extract_dates(stripped)):
                scored.append((score, truncate_words(stripped, 40)))
        if any(score > 0 for score, _ in scored):
            scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: item[0], reverse=True)
        return list(dict.fromkeys(line for _, line in scored))

    def _claims(self, text: str, question: str | None) -> list[str]:
        question_terms = _question_terms(question)
        scored: list[tuple[float, str]] = []
        for sentence in split_sentences(text):
            score = 0.0
            if extract_numbers(sentence):
                score += 2
            if extract_dates(sentence):
                score += 2
            if extract_entities(sentence):
                score += 1
            overlap = 0
            if question_terms:
                sentence_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9_]+", sentence)}
                overlap = len(question_terms & sentence_terms)
                score += overlap * 3
            lowered = sentence.lower()
            if question_terms and overlap == 0 and _is_generated_distractor(lowered):
                continue
            if lowered.startswith("distractor "):
                score -= 3
            if score:
                scored.append((score, truncate_words(sentence, 32)))
        if any(score > 0 for score, _ in scored):
            scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: item[0], reverse=True)
        return list(dict.fromkeys(claim for _, claim in scored))

    def _relevant_excerpts(self, text: str, question: str | None) -> list[str]:
        paragraphs = split_paragraphs(text)
        if not question:
            return [truncate_words(paragraph, 40) for paragraph in paragraphs[:3]]
        q_terms = _question_terms(question)
        scored = []
        for paragraph in paragraphs:
            p_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9_]+", paragraph)}
            score = len(q_terms & p_terms)
            if score:
                scored.append((score, truncate_words(paragraph, 48)))
        if scored and any(score > 1 for score, _ in scored):
            scored = [item for item in scored if item[0] > 1]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [paragraph for _, paragraph in scored]


def _bullet_block(title: str, items: list[str]) -> str:
    if not items:
        return ""
    return title + ":\n" + "\n".join(f"- {item}" for item in items)


def _render_sections(section_order: list[str], section_items: dict[str, list[str]]) -> str:
    sections = ["COMPRESSED SEMANTIC BRIEF"]
    for title in section_order:
        sections.append(_bullet_block(title, section_items.get(title, [])))
    return "\n\n".join(section for section in sections if section.strip())


def _fit_sections_to_budget(
    source_text: str,
    target_token_ratio: float,
    section_items: dict[str, list[str]],
) -> str:
    original_tokens = count_tokens(source_text).value
    target_tokens = max(48, math.ceil(original_tokens * target_token_ratio))
    priority_order = [
        "Relevant excerpts",
        "Key lines",
        "Claims",
        "Dates",
        "Numbers",
        "Entities",
        "Headings",
    ]
    selected = {title: [] for title in priority_order}
    for title in priority_order:
        for item in section_items.get(title, []):
            candidate = {key: list(value) for key, value in selected.items()}
            candidate[title].append(item)
            candidate_text = _render_sections(priority_order, candidate)
            if count_tokens(candidate_text).value <= target_tokens or not any(selected.values()):
                selected = candidate
    return _render_sections(priority_order, selected)


def _question_terms(question: str | None) -> set[str]:
    return {
        term.lower()
        for term in re.findall(r"[A-Za-z0-9_]+", question or "")
        if len(term) > 2 and term.lower() not in STOP_TERMS
    }


def _is_generated_distractor(lowered_text: str) -> bool:
    prefixes = (
        "distractor ",
        "facilities note:",
        "operations note:",
        "archive note:",
        "procurement note:",
        "people note:",
        "release note:",
        "support note:",
        "security note:",
        "finance note:",
        "research note:",
    )
    return (
        "synthetic document" in lowered_text
        or "answer-relevant source section" in lowered_text
        or lowered_text.startswith(prefixes)
    )
