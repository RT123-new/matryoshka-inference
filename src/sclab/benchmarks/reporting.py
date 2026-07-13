from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from sclab.utils.jsonl import read_jsonl


def generate_report(run_dir: str | Path) -> tuple[Path, Path]:
    root = Path(run_dir)
    results = list(read_jsonl(root / "results.jsonl"))
    report = build_markdown_report(results, root)
    failures = build_failures_markdown(results)
    report_path = root / "report.md"
    failures_path = root / "failures.md"
    report_path.write_text(report, encoding="utf-8")
    failures_path.write_text(failures, encoding="utf-8")
    return report_path, failures_path


def build_markdown_report(results: list[dict[str, Any]], run_dir: Path) -> str:
    by_compressor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in results:
        by_compressor[str(record.get("compressor"))].append(record)
        by_type[str(record.get("task_type"))].append(record)

    lines = [
        "# Semantic Compression Benchmark Report",
        "",
        "## Executive summary",
        "",
        _executive_summary(results),
        "",
        "## Hardware profile",
        "",
        _hardware_summary(run_dir / "hardware.json"),
        "",
        "## Runtime/model details",
        "",
        _runtime_summary(results),
        "",
        "## Compressor leaderboard",
        "",
        "| compressor | tasks | avg quality | avg prompt ratio | avg latency factor | pass rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    ranked = sorted(by_compressor.items(), key=lambda item: _avg(item[1], "quality_score"), reverse=True)
    for compressor, rows in ranked:
        lines.append(
            f"| {compressor} | {len(rows)} | {_avg(rows, 'quality_score'):.3f} | "
            f"{_avg(rows, 'compression_ratio'):.3f} | {_avg(rows, 'latency_improvement_factor'):.3f} | "
            f"{_pass_rate(rows):.1%} |"
        )

    lines.extend(
        [
            "",
            "## Compression vs quality table",
            "",
            "| compressor | avg compressed prompt tokens | avg raw prompt tokens | avg quality |",
            "|---|---:|---:|---:|",
        ]
    )
    for compressor, rows in sorted(by_compressor.items()):
        lines.append(
            f"| {compressor} | {_avg(rows, 'compressed_prompt_tokens'):.1f} | "
            f"{_avg(rows, 'raw_prompt_tokens'):.1f} | {_avg(rows, 'quality_score'):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Compression vs latency table",
            "",
            "| compressor | avg raw total s | avg compressed total s | avg TTFT s | avg decode tok/s |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for compressor, rows in sorted(by_compressor.items()):
        lines.append(
            f"| {compressor} | {_avg(rows, 'raw_total_time_s'):.4f} | "
            f"{_avg(rows, 'compressed_total_time_s'):.4f} | {_avg(rows, 'time_to_first_token_s'):.4f} | "
            f"{_avg(rows, 'decode_tokens_per_s'):.2f} |"
        )

    lines.extend(["", "## Best task types", "", "| task type | avg quality | pass rate |", "|---|---:|---:|"])
    for task_type, rows in sorted(by_type.items(), key=lambda item: _avg(item[1], "quality_score"), reverse=True):
        lines.append(f"| {task_type} | {_avg(rows, 'quality_score'):.3f} | {_pass_rate(rows):.1%} |")

    lines.extend(["", "## Worst task types", "", "| task type | avg quality | pass rate |", "|---|---:|---:|"])
    for task_type, rows in sorted(by_type.items(), key=lambda item: _avg(item[1], "quality_score")):
        lines.append(f"| {task_type} | {_avg(rows, 'quality_score'):.3f} | {_pass_rate(rows):.1%} |")

    lines.extend(["", "## Failure examples", ""])
    failures = [record for record in results if record.get("failure_reasons")]
    if not failures:
        lines.append("No scored failures were recorded.")
    else:
        for record in failures[:10]:
            reasons = ", ".join(record.get("failure_reasons", []))
            lines.append(
                f"- `{record.get('task_id')}` / `{record.get('compressor')}`: {reasons}. "
                f"Answer: {record.get('answer', '')[:180]}"
            )

    lines.extend(["", "## Recommendation", "", _recommendation(results), ""])
    return "\n".join(lines)


def build_failures_markdown(results: list[dict[str, Any]]) -> str:
    lines = ["# Failure Examples", ""]
    failures = [record for record in results if record.get("failure_reasons")]
    if not failures:
        lines.append("No failures recorded.")
        return "\n".join(lines)
    for record in failures:
        lines.extend(
            [
                f"## {record.get('task_id')} / {record.get('compressor')}",
                "",
                f"- Reasons: {', '.join(record.get('failure_reasons', []))}",
                f"- Quality: {record.get('quality_score')}",
                f"- Compression ratio: {record.get('compression_ratio')}",
                f"- Raw total time: {record.get('raw_total_time_s')}",
                f"- Compressed total time: {record.get('compressed_total_time_s')}",
                "",
                "Answer:",
                "",
                str(record.get("answer", "")),
                "",
                "Gold:",
                "",
                str(record.get("gold_answer", "")),
                "",
            ]
        )
    return "\n".join(lines)


def _executive_summary(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No results were produced."
    best = max(
        set(record.get("compressor") for record in results),
        key=lambda name: _avg([row for row in results if row.get("compressor") == name], "quality_score"),
    )
    avg_quality = _avg(results, "quality_score")
    avg_ratio = _avg(results, "compression_ratio")
    avg_latency = _avg(results, "latency_improvement_factor")
    fake = any(record.get("runtime") == "fake" for record in results)
    warning = " This run used the fake runtime, so timing is only for harness validation." if fake else ""
    return (
        f"Average quality was {avg_quality:.3f}, average prompt ratio was {avg_ratio:.3f}, "
        f"and average latency factor was {avg_latency:.3f}. Best quality compressor: `{best}`.{warning}"
    )


def _hardware_summary(path: Path) -> str:
    if not path.exists():
        return "Hardware profile was not found."
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"Hardware profile could not be parsed: {exc}"
    platform_data = data.get("platform", {})
    memory = data.get("memory", {})
    runtimes = data.get("runtimes", {})
    total_gb = memory.get("total_bytes") or memory.get("sysctl_total_bytes")
    total_text = f"{total_gb / (1024 ** 3):.1f} GB" if isinstance(total_gb, int) else "unknown"
    return (
        f"- System: {platform_data.get('system')} {platform_data.get('release')} on {platform_data.get('machine')}\n"
        f"- Chip/CPU: {platform_data.get('chip_name') or platform_data.get('processor') or 'unknown'}\n"
        f"- Memory: {total_text}\n"
        f"- Ollama: {runtimes.get('ollama') or 'not found'}\n"
        f"- MLX: {runtimes.get('mlx_lm_generate') or 'not found'}\n"
        f"- llama.cpp: {runtimes.get('llama_cli') or 'not found'}"
    )


def _runtime_summary(results: list[dict[str, Any]]) -> str:
    pairs = sorted({(record.get("runtime"), record.get("model")) for record in results})
    return "\n".join(f"- Runtime `{runtime}`, model `{model}`" for runtime, model in pairs)


def _recommendation(results: list[dict[str, Any]]) -> str:
    real_rows = [record for record in results if record.get("compressor") not in {"raw", "oracle", "gzip_b64_control"}]
    if not real_rows:
        return "Conclusion: run real compressors before deciding whether to continue."
    compressors = sorted({str(record.get("compressor")) for record in real_rows})
    best_name = max(
        compressors,
        key=lambda name: _avg([row for row in real_rows if row.get("compressor") == name], "utility_score"),
    )
    best_rows = [row for row in real_rows if row.get("compressor") == best_name]
    quality = _avg(best_rows, "quality_score")
    avg_ratio = _avg(best_rows, "compression_ratio")
    ratio_reduction = 1.0 - avg_ratio
    latency = _avg(best_rows, "latency_improvement_factor")
    if avg_ratio >= 1.0:
        return (
            f"Conclusion: `{best_name}` did not reduce full prompt size after template overhead. "
            "Use longer documents, lower budgets, or more compact prompt templates before drawing speed conclusions."
        )
    if quality >= 0.85 and ratio_reduction >= 0.40 and latency >= 1.15:
        return (
            f"Conclusion: `{best_name}` is promising for this setup. "
            "Continue, but inspect failures before using it for legal, numeric, or technical exactness."
        )
    if quality < 0.75:
        return (
            f"Conclusion: `{best_name}` quality loss is too high. Modify the compressor to preserve raw excerpts, "
            "especially numbers, dates, contradictions, and code/config details."
        )
    if latency < 1.05:
        return (
            f"Conclusion: `{best_name}` reduces prompt size, but speed gains are weak here. "
            "Measure prompt eval and decode separately before moving to deeper model-level experiments."
        )
    return f"Conclusion: `{best_name}` results are mixed. Continue with ablations to find the quality cliff."


def _avg(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else 0.0


def _pass_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get("passed")) / len(rows)
