from __future__ import annotations

import math
import re

from sclab.compressors.base import Document, build_result
from sclab.tokenization import count_tokens
from sclab.utils.numbers_dates import extract_dates, extract_numbers
from sclab.utils.text_split import split_paragraphs


class ExtractiveRelevanceCompressor:
    name = "extractive_relevance"

    def __init__(self, target_token_ratio: float = 0.35, max_paragraphs: int = 8) -> None:
        self.target_token_ratio = target_token_ratio
        self.max_paragraphs = max_paragraphs

    def compress(self, doc: Document, question: str | None = None):
        paragraphs = split_paragraphs(doc.text)
        original_tokens = count_tokens(doc.text).value
        budget = max(32, math.ceil(original_tokens * self.target_token_ratio))
        if not paragraphs:
            selected = [doc.text]
            selected_indexes = [0]
        else:
            scores = self._score_paragraphs(paragraphs, question)
            ranked = sorted(range(len(paragraphs)), key=lambda idx: scores[idx], reverse=True)
            positive_ranked = [idx for idx in ranked if scores[idx] > 0]
            if positive_ranked:
                ranked = positive_ranked
            selected_indexes = []
            used = 0
            for idx in ranked:
                para_tokens = count_tokens(paragraphs[idx]).value
                if selected_indexes and used + para_tokens > budget:
                    continue
                selected_indexes.append(idx)
                used += para_tokens
                if used >= budget or len(selected_indexes) >= self.max_paragraphs:
                    break
            if not selected_indexes:
                selected_indexes = ranked[:1]
            selected_indexes.sort()
            selected = [paragraphs[idx] for idx in selected_indexes]
        text = "\n\n".join(selected)
        return build_result(
            doc=doc,
            text=text,
            method=self.name,
            metadata={
                "target_token_ratio": self.target_token_ratio,
                "selected_paragraph_indexes": selected_indexes,
                "paragraph_count": len(paragraphs),
                "preserves_source_text": True,
            },
        )

    def _score_paragraphs(self, paragraphs: list[str], question: str | None) -> list[float]:
        if not question:
            return [1.0 / (idx + 1) for idx, _ in enumerate(paragraphs)]
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform([question, *paragraphs])
            similarities = cosine_similarity(matrix[0], matrix[1:]).flatten()
            return [
                float(value) + _intent_bonus(question, paragraph)
                for value, paragraph in zip(similarities, paragraphs, strict=True)
            ]
        except Exception:
            q_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9_]+", question) if len(term) > 2}
            scores = []
            for idx, paragraph in enumerate(paragraphs):
                p_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9_]+", paragraph)}
                overlap = len(q_terms & p_terms)
                scores.append(overlap + _intent_bonus(question, paragraph) + (0.01 / (idx + 1)))
            return scores


def _intent_bonus(question: str, paragraph: str) -> float:
    q = question.lower()
    p = paragraph.lower()
    score = 0.0
    if _has_any(q, {"obligation", "obligations", "required", "requirement"}) and _has_any(
        p,
        {"must", "shall", "required", "maintain", "deliver", "pay", "within", "every"},
    ):
        score += 1.5
    if _has_any(q, {"deadline", "deadlines", "threshold", "thresholds", "date", "dates"}) and (
        extract_numbers(paragraph) or extract_dates(paragraph) or _has_any(p, {"day", "days", "friday", "month", "%"})
    ):
        score += 1.0
    if _has_any(q, {"cheaper", "cost", "costs", "setup", "fee", "fees"}) and _has_any(
        p,
        {"option", "cost", "costs", "setup", "fee", "fees", "month"},
    ):
        score += 1.0
    if _has_any(q, {"contradiction", "conflict", "discrepancy"}) and _has_any(
        p,
        {"but", "conflict", "discrepancy", "not", "requires", "supports", "resolved"},
    ):
        score += 1.0
    return score


def _has_any(text: str, terms: set[str]) -> bool:
    words = set(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()))
    return any(term in words if re.search(r"[a-z0-9]", term) else term in text for term in terms)
