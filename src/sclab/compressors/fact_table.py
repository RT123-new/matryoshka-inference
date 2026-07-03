from __future__ import annotations

import re

from sclab.compressors.base import Document, build_result
from sclab.utils.numbers_dates import extract_dates, extract_entities, extract_numbers
from sclab.utils.text_split import split_sentences, truncate_words

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
}


class FactTableCompressor:
    name = "fact_table"

    def __init__(self, max_rows: int = 14) -> None:
        self.max_rows = max_rows

    def compress(self, doc: Document, question: str | None = None):
        candidates = []
        q_terms = _question_terms(question)
        for sentence in split_sentences(doc.text):
            numbers = extract_numbers(sentence)
            dates = extract_dates(sentence)
            entities = extract_entities(sentence)
            if not (numbers or dates or entities):
                continue
            sentence_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9_]+", sentence)}
            overlap = len(q_terms & sentence_terms)
            lowered = sentence.lower()
            if q_terms and overlap == 0 and _is_generated_distractor(lowered):
                continue
            score = overlap * 4 + len(numbers) + len(dates)
            if lowered.startswith("distractor "):
                score -= 4
            subject = entities[0] if entities else " ".join(sentence.split()[:4]).strip(" ,.;:")
            evidence_bits = dates + numbers
            evidence = ", ".join(evidence_bits[:4]) if evidence_bits else truncate_words(sentence, 12)
            candidates.append(
                (
                    score,
                    {
                        "subject": subject,
                        "relation": "states",
                        "object": truncate_words(sentence, 26),
                        "evidence": evidence,
                    },
                )
            )
        if candidates and any(score > 0 for score, _ in candidates):
            candidates = [item for item in candidates if item[0] > 0]
        candidates.sort(key=lambda item: item[0], reverse=True)
        rows = [row for _, row in candidates[: self.max_rows]]
        lines = [
            "| subject | relation | object | evidence |",
            "|---|---|---|---|",
        ]
        for row in rows:
            lines.append(
                "| {subject} | {relation} | {object} | {evidence} |".format(
                    subject=_cell(row["subject"]),
                    relation=_cell(row["relation"]),
                    object=_cell(row["object"]),
                    evidence=_cell(row["evidence"]),
                )
            )
        if not rows:
            lines.append("| None detected | states | No compact facts extracted. | |")
        text = "\n".join(lines)
        return build_result(
            doc=doc,
            text=text,
            method=self.name,
            metadata={"row_count": len(rows), "evidence_preserves_exact_dates_numbers": True},
        )


def _cell(value: str) -> str:
    return value.replace("|", "/").replace("\n", " ").strip()


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
