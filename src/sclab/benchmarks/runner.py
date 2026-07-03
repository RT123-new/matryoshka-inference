from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sclab.benchmarks.datasets import BenchmarkTask, load_tasks
from sclab.benchmarks.reporting import generate_report
from sclab.benchmarks.scoring import ScoreResult, score_answer
from sclab.compressors import Document, get_compressor
from sclab.hardware import scan_hardware
from sclab.runtimes.base import GenerationRequest, GenerationResult
from sclab.tokenization import count_tokens
from sclab.utils.jsonl import write_jsonl

RAW_PROMPT = """You are answering from the provided source text.

Rules:
- Answer only from the source.
- If the source does not contain the answer, say "Not found in source."
- Preserve exact numbers, dates, names, and conditions.
- Be concise.

Source:
{context}

Question:
{question}

Answer:
"""

COMPRESSED_PROMPT = """You are answering from a compressed representation of the source text.

Rules:
- Answer only from the compressed source.
- If the compressed source does not contain the answer, say "Not found in compressed source."
- Preserve exact numbers, dates, names, and conditions.
- Be concise.
- If the compression appears insufficient, say what is missing.

Compressed source:
{compressed_context}

Question:
{question}

Answer:
"""

HYBRID_PROMPT = """You are answering from a compressed source that includes extracted facts and selected exact excerpts.

Rules:
- Prefer exact excerpts for numbers, dates, names, commands, and legal/technical wording.
- Use the semantic brief only for orientation.
- If evidence is missing, say "Not found in compressed source."

Compressed source:
{compressed_context}

Question:
{question}

Answer:
"""


@dataclass
class BenchmarkConfig:
    runtime: str
    model: str
    dataset: str
    compressors: list[str]
    out: str
    max_tasks: int | None = None
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    timeout_s: int = 300
    budget: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    runtime_options: dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    def __init__(self, runtime) -> None:
        self.runtime = runtime

    def run(self, config: BenchmarkConfig) -> list[dict[str, Any]]:
        run_dir = Path(config.out)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "compression_examples").mkdir(exist_ok=True)
        (run_dir / "plots").mkdir(exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8")
        (run_dir / "hardware.json").write_text(json.dumps(scan_hardware(), indent=2, sort_keys=True), encoding="utf-8")

        tasks = load_tasks(config.dataset, config.max_tasks)
        results: list[dict[str, Any]] = []
        for task in tasks:
            results.extend(self._run_task(task, config, run_dir))

        write_jsonl(run_dir / "results.jsonl", results)
        generate_report(run_dir)
        return results

    def _run_task(self, task: BenchmarkTask, config: BenchmarkConfig, run_dir: Path) -> list[dict[str, Any]]:
        doc = Document(
            text=task.document,
            id=task.id,
            metadata={"source_span": task.source_span, **task.metadata},
        )
        raw_prompt = render_prompt("raw", task.document, task.question)
        raw_result = self._generate(task, config, raw_prompt)
        raw_prompt_tokens = raw_result.prompt_tokens or self.runtime.count_tokens(raw_prompt, config.model)
        if raw_result.raw_metadata.get("error"):
            raw_score = ScoreResult(0, False, 0, False, 0, 0, [raw_result.raw_metadata["error"]])
        else:
            raw_score = score_answer(raw_result.text, task)

        rows = []
        for compressor_name in config.compressors:
            try:
                compressor = get_compressor(compressor_name, budget=config.budget)
                compression = compressor.compress(doc, task.question)
                prompt = render_prompt(compressor.name, compression.compressed_text, task.question)
                if compressor.name == "raw":
                    result = raw_result
                    score = raw_score
                    prompt_tokens = raw_prompt_tokens
                else:
                    result = self._generate(task, config, prompt)
                    prompt_tokens = result.prompt_tokens or self.runtime.count_tokens(prompt, config.model)
                    if result.raw_metadata.get("error"):
                        score = ScoreResult(0, False, 0, False, 0, 0, [result.raw_metadata["error"]])
                    else:
                        score = score_answer(result.text, task)
                ratio = prompt_tokens / raw_prompt_tokens if raw_prompt_tokens else None
                latency_factor = raw_result.total_time_s / result.total_time_s if result.total_time_s else None
                failure_reasons = list(score.failure_reasons)
                if result.raw_metadata.get("error"):
                    failure_reasons.append("runtime_error")
                passed = bool(
                    score.quality_score >= 0.90
                    and result.total_time_s <= raw_result.total_time_s * 0.90
                    and prompt_tokens <= raw_prompt_tokens * 0.70
                    and not result.raw_metadata.get("error")
                )
                if not passed and not failure_reasons:
                    failure_reasons.extend(_pass_fail_reasons(score.quality_score, result.total_time_s, raw_result.total_time_s, prompt_tokens, raw_prompt_tokens))
                record = {
                    "task_id": task.id,
                    "task_type": task.type,
                    "difficulty": task.difficulty,
                    "runtime": self.runtime.name,
                    "model": config.model,
                    "compressor": compressor.name,
                    "raw_prompt_tokens": raw_prompt_tokens,
                    "compressed_prompt_tokens": prompt_tokens,
                    "context_original_tokens": compression.original_tokens,
                    "context_compressed_tokens": compression.compressed_tokens,
                    "compression_ratio": ratio,
                    "context_compression_ratio": compression.compression_ratio_tokens,
                    "raw_total_time_s": raw_result.total_time_s,
                    "compressed_total_time_s": result.total_time_s,
                    "latency_improvement_factor": latency_factor,
                    "time_to_first_token_s": result.time_to_first_token_s,
                    "prompt_eval_time_s": result.prompt_eval_time_s,
                    "decode_time_s": result.decode_time_s,
                    "completion_tokens": result.completion_tokens,
                    "decode_tokens_per_s": result.decode_tokens_per_s,
                    "quality_score": score.quality_score,
                    "utility_score": _utility_score(score.quality_score, latency_factor, ratio),
                    "passed": passed,
                    "answer": result.text,
                    "gold_answer": task.gold_answer,
                    "failure_reasons": list(dict.fromkeys(failure_reasons)),
                    "score_details": score.to_dict(),
                    "compression": compression.to_dict(),
                    "runtime_metadata": result.raw_metadata,
                    "created_at": int(time.time()),
                }
                rows.append(record)
                self._write_example(run_dir, task, compressor.name, compression.compressed_text)
            except Exception as exc:
                rows.append(_exception_record(task, config, self.runtime.name, compressor_name, raw_result, raw_prompt_tokens, exc))
        return rows

    def _generate(self, task: BenchmarkTask, config: BenchmarkConfig, prompt: str) -> GenerationResult:
        options: dict[str, Any] = dict(config.runtime_options)
        if self.runtime.name == "fake":
            options["task"] = task.to_dict()
        return self.runtime.generate(
            GenerationRequest(
                model=config.model,
                prompt=prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                seed=config.seed,
                timeout_s=config.timeout_s,
                runtime_options=options,
            )
        )

    def _write_example(self, run_dir: Path, task: BenchmarkTask, compressor_name: str, compressed_text: str) -> None:
        safe_name = f"{task.id}_{compressor_name}".replace("/", "_")
        path = run_dir / "compression_examples" / f"{safe_name}.txt"
        if path.exists():
            return
        path.write_text(
            f"Task: {task.id}\nQuestion: {task.question}\nCompressor: {compressor_name}\n\n{compressed_text}\n",
            encoding="utf-8",
        )


def render_prompt(compressor_name: str, context: str, question: str) -> str:
    if compressor_name == "raw":
        return RAW_PROMPT.format(context=context, question=question)
    if compressor_name == "hybrid_brief_excerpts":
        return HYBRID_PROMPT.format(compressed_context=context, question=question)
    return COMPRESSED_PROMPT.format(compressed_context=context, question=question)


def _utility_score(quality_score: float, latency_factor: float | None, compression_ratio: float | None) -> float:
    if quality_score < 0.75:
        return 0.0
    latency = min(latency_factor or 0.0, 3.0)
    safety = 1.0 if (compression_ratio or 1.0) <= 0.70 else 0.75
    return round(quality_score * latency * safety, 4)


def _pass_fail_reasons(
    quality: float,
    compressed_time: float,
    raw_time: float,
    compressed_tokens: int,
    raw_tokens: int,
) -> list[str]:
    reasons = []
    if quality < 0.90:
        reasons.append("quality_below_0.90")
    if compressed_time > raw_time * 0.90:
        reasons.append("latency_not_10_percent_faster")
    if compressed_tokens > raw_tokens * 0.70:
        reasons.append("prompt_not_reduced_30_percent")
    return reasons


def _exception_record(
    task: BenchmarkTask,
    config: BenchmarkConfig,
    runtime_name: str,
    compressor_name: str,
    raw_result: GenerationResult,
    raw_prompt_tokens: int,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "task_id": task.id,
        "task_type": task.type,
        "difficulty": task.difficulty,
        "runtime": runtime_name,
        "model": config.model,
        "compressor": compressor_name,
        "raw_prompt_tokens": raw_prompt_tokens,
        "compressed_prompt_tokens": None,
        "compression_ratio": None,
        "raw_total_time_s": raw_result.total_time_s,
        "compressed_total_time_s": 0,
        "time_to_first_token_s": None,
        "prompt_eval_time_s": None,
        "decode_time_s": None,
        "completion_tokens": None,
        "decode_tokens_per_s": None,
        "quality_score": 0,
        "utility_score": 0,
        "passed": False,
        "answer": "",
        "gold_answer": task.gold_answer,
        "failure_reasons": [f"exception: {type(exc).__name__}: {exc}"],
        "score_details": {},
        "compression": {},
        "runtime_metadata": {},
        "created_at": int(time.time()),
    }
