from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from sclab.benchmarks.datasets import BenchmarkTask
from sclab.utils.numbers_dates import extract_dates, extract_numbers, normalize_number


@dataclass
class ScoreResult:
    quality_score: float
    exact_or_alias_match: bool
    must_include_coverage: float
    must_not_include_violation: bool
    numeric_date_coverage: float
    answer_length: int
    failure_reasons: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def score_answer(answer: str, task: BenchmarkTask) -> ScoreResult:
    normalized_answer = _normalize(answer)
    normalized_gold = _normalize(task.gold_answer)
    failure_reasons: list[str] = []

    if not task.gold_answer:
        is_not_found = "not found" in normalized_answer
        return ScoreResult(
            quality_score=1.0 if is_not_found else 0.0,
            exact_or_alias_match=is_not_found,
            must_include_coverage=1.0 if is_not_found else 0.0,
            must_not_include_violation=False,
            numeric_date_coverage=1.0,
            answer_length=len(answer.split()),
            failure_reasons=[] if is_not_found else ["no_gold_answer_for_automatic_scoring"],
        )

    aliases = [_normalize(alias) for alias in task.answer_aliases]
    exact_or_alias = normalized_gold in normalized_answer or any(alias and alias in normalized_answer for alias in aliases)

    must_include = [_normalize(item) for item in task.must_include]
    if must_include:
        included = sum(1 for item in must_include if _contains_required(normalized_answer, item, answer))
        must_include_coverage = included / len(must_include)
    else:
        must_include_coverage = 1.0 if exact_or_alias else 0.0
    if must_include_coverage < 1.0:
        failure_reasons.append("must_include_missing")

    must_not = [_normalize(item) for item in task.must_not_include]
    violation = any(item and item in normalized_answer for item in must_not)
    if violation:
        failure_reasons.append("must_not_include_violation")

    expected_numbers = [normalize_number(value) for value in extract_numbers(task.gold_answer)]
    expected_dates = [_normalize(value) for value in extract_dates(task.gold_answer)]
    answer_numbers = {normalize_number(value) for value in extract_numbers(answer)}
    answer_dates = {_normalize(value) for value in extract_dates(answer)}
    required_precision = expected_numbers + expected_dates
    if required_precision:
        covered = 0
        for value in expected_numbers:
            if value in answer_numbers or value in normalized_answer:
                covered += 1
        for value in expected_dates:
            if value in answer_dates or value in normalized_answer:
                covered += 1
        numeric_date_coverage = covered / len(required_precision)
    else:
        numeric_date_coverage = 1.0
    if numeric_date_coverage < 1.0:
        failure_reasons.append("numeric_or_date_mismatch")

    evidence_match = must_include_coverage >= 1.0 and numeric_date_coverage >= 1.0 and not violation
    exact_component = 1.0 if exact_or_alias else (0.8 if evidence_match and task.must_include else 0.0)
    if not exact_or_alias and exact_component == 0.0:
        failure_reasons.append("exact_or_alias_missing")

    score = (
        0.35 * exact_component
        + 0.35 * must_include_coverage
        + 0.20 * numeric_date_coverage
        + 0.10 * (0.0 if violation else 1.0)
    )
    if "not found" in normalized_answer and "not found" not in normalized_gold:
        score = min(score, 0.35)
        failure_reasons.append("incorrect_not_found")
    return ScoreResult(
        quality_score=round(max(0.0, min(1.0, score)), 4),
        exact_or_alias_match=exact_or_alias or evidence_match,
        must_include_coverage=round(must_include_coverage, 4),
        must_not_include_violation=violation,
        numeric_date_coverage=round(numeric_date_coverage, 4),
        answer_length=len(answer.split()),
        failure_reasons=list(dict.fromkeys(failure_reasons)),
    )


def _normalize(text: str) -> str:
    lowered = text.lower().strip()
    return re.sub(r"\s+", " ", lowered)


def _contains_required(normalized_answer: str, normalized_item: str, original_answer: str) -> bool:
    if normalized_item in normalized_answer:
        return True
    compact_answer = re.sub(r"[\s,£$]", "", normalized_answer)
    compact_item = re.sub(r"[\s,£$]", "", normalized_item)
    if compact_item and compact_item in compact_answer:
        return True
    item_numbers = [normalize_number(value) for value in extract_numbers(normalized_item)]
    if item_numbers:
        answer_numbers = {normalize_number(value) for value in extract_numbers(original_answer)}
        return all(value in answer_numbers or value in compact_answer for value in item_numbers)
    return False
