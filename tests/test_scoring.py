from sclab.benchmarks.datasets import BenchmarkTask
from sclab.benchmarks.scoring import score_answer


def test_score_exact_with_numbers():
    task = BenchmarkTask(
        id="t1",
        type="number_date_precision",
        document="Amount due: GBP 2,487.65. Due date: 2026-09-30.",
        question="What amount and date?",
        gold_answer="GBP 2,487.65 due on 2026-09-30",
        must_include=["2,487.65", "2026-09-30"],
    )
    score = score_answer("GBP 2,487.65 due on 2026-09-30", task)
    assert score.quality_score == 1.0


def test_score_penalizes_forbidden_answer():
    task = BenchmarkTask(
        id="t2",
        type="contradiction_detection",
        document="A says 30 days. B says 14 days.",
        question="Contradiction?",
        gold_answer="Yes, 30 days conflicts with 14 days.",
        must_include=["30 days", "14 days"],
        must_not_include=["no contradiction"],
    )
    score = score_answer("There is no contradiction.", task)
    assert score.quality_score < 0.75
    assert "must_not_include_violation" in score.failure_reasons


def test_score_accepts_paraphrase_with_required_evidence():
    task = BenchmarkTask(
        id="t3",
        type="multi_fact",
        document="",
        question="Compare options",
        gold_answer="Option B is cheaper over 12 months: GBP 1,140 versus GBP 1,360 for Option A.",
        must_include=["Option B", "1,140", "1,360"],
    )
    answer = "Over 12 months, Option B costs GBP 1140, while Option A costs GBP 1360, so Option B is cheaper."
    score = score_answer(answer, task)
    assert score.quality_score >= 0.9
    assert score.failure_reasons == []
