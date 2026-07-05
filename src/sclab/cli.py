from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sclab.benchmarks.datasets import BenchmarkTask, ingest_documents, write_tasks
from sclab.benchmarks.reporting import generate_report
from sclab.benchmarks.runner import BenchmarkConfig, BenchmarkRunner
from sclab.compressors import Document, compressor_names, get_compressor
from sclab.hardware import write_hardware_profile
from sclab.runtimes import available_runtime_names, get_runtime
from sclab.runtimes.base import GenerationRequest
from sclab.tokenization import count_tokens


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sclab", description="Semantic Compression Lab CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Detect hardware and local runtimes")
    scan.add_argument("--out", default="hardware.json", help="Path to write hardware JSON")
    scan.set_defaults(func=cmd_scan)

    benchmark = subparsers.add_parser("benchmark", help="Run a compression benchmark")
    benchmark.add_argument("--runtime", required=True, choices=available_runtime_names())
    benchmark.add_argument("--model", required=True)
    benchmark.add_argument("--dataset", required=True)
    benchmark.add_argument("--compressors", required=True, help="Comma-separated compressor names")
    benchmark.add_argument("--max-tasks", type=int)
    benchmark.add_argument("--max-tokens", type=int, default=256)
    benchmark.add_argument("--temperature", type=float, default=0.0)
    benchmark.add_argument("--top-p", type=float, default=1.0)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--timeout-s", type=int, default=300)
    benchmark.add_argument(
        "--runtime-options",
        default=None,
        help='JSON dict of runtime options, e.g. \'{"mode":"diffusion","block_size":8,"adaptive":true,"copy":true}\'',
    )
    benchmark.add_argument("--out", required=True)
    benchmark.set_defaults(func=cmd_benchmark)

    compare = subparsers.add_parser("compare", help="Generate Markdown report from a run directory")
    compare.add_argument("run_dir")
    compare.set_defaults(func=cmd_compare)

    single = subparsers.add_parser("single", help="Ask one question against one local document")
    single.add_argument("--runtime", required=True, choices=available_runtime_names())
    single.add_argument("--model", required=True)
    single.add_argument("--document", required=True)
    single.add_argument("--question", required=True)
    single.add_argument("--compressor", required=True, choices=compressor_names())
    single.add_argument("--max-tokens", type=int, default=256)
    single.add_argument("--temperature", type=float, default=0.0)
    single.add_argument("--out")
    single.set_defaults(func=cmd_single)

    ablate = subparsers.add_parser("ablate", help="Run one compressor across token budgets")
    ablate.add_argument("--runtime", required=True, choices=available_runtime_names())
    ablate.add_argument("--model", required=True)
    ablate.add_argument("--dataset", required=True)
    ablate.add_argument("--compressor", required=True, choices=compressor_names())
    ablate.add_argument("--budgets", required=True, help="Comma-separated ratios, e.g. 0.2,0.4,0.6")
    ablate.add_argument("--max-tasks", type=int)
    ablate.add_argument("--max-tokens", type=int, default=256)
    ablate.add_argument("--temperature", type=float, default=0.0)
    ablate.add_argument("--out", default=None)
    ablate.set_defaults(func=cmd_ablate)

    ingest = subparsers.add_parser("ingest", help="Ingest local text/code/config files into manual tasks")
    ingest.add_argument("path")
    ingest.add_argument("--out", required=True)
    ingest.set_defaults(func=cmd_ingest)

    compress = subparsers.add_parser(
        "compress", help="Compress a document for a question and print the result (pipeable)"
    )
    compress.add_argument("--document", required=True)
    compress.add_argument("--question", required=True)
    compress.add_argument("--compressor", default="extractive_relevance", choices=compressor_names())
    compress.add_argument("--budget", type=float, default=None, help="Target ratio, e.g. 0.3")
    compress.add_argument("--stats", action="store_true", help="Print token stats to stderr")
    compress.set_defaults(func=cmd_compress)

    for verb, helptext in (("serve", "OpenAI-compatible server + live dashboard"),
                           ("up", "Start the server AND open the dashboard (one command)")):
        s = subparsers.add_parser(verb, help=helptext)
        s.add_argument("--model", default=None,
                       help="Orthrus alias (orthrus-qwen3-1.7b/4b/8b), HF repo id, "
                            "or an upstream model name when --upstream/--backend proxy is used")
        s.add_argument("--backend", default="auto", choices=["auto", "orthrus", "proxy"],
                       help="orthrus = accelerated MLX; proxy = forward to any OpenAI-compatible model")
        s.add_argument("--upstream", default=None,
                       help="Upstream base URL for proxy mode, e.g. http://localhost:11434/v1 (Ollama)")
        s.add_argument("--api-key", default="", help="Bearer token for the upstream (if any)")
        s.add_argument("--host", default="127.0.0.1")
        s.add_argument("--port", type=int, default=8977)
        s.add_argument("--mode", default="auto", choices=["auto", "diffusion", "ar"])
        s.add_argument("--block-size", type=int, default=16)
        s.add_argument("--backoff", type=int, default=96)
        s.add_argument("--served-name", default=None, help="Model name to report to clients")
        s.add_argument("--open", dest="open_dash", action="store_true", help="Open the dashboard in a browser")
        s.set_defaults(func=cmd_serve, _is_up=(verb == "up"))

    hcfg = subparsers.add_parser("hermes-config", help="Print the exact Hermes provider settings to paste")
    hcfg.add_argument("--host", default="127.0.0.1")
    hcfg.add_argument("--port", type=int, default=8977)
    hcfg.add_argument("--model", default="orthrus-qwen3-4b")
    hcfg.set_defaults(func=cmd_hermes_config)

    return parser


def cmd_scan(args: argparse.Namespace) -> int:
    profile = write_hardware_profile(args.out)
    print(json.dumps(profile, indent=2, sort_keys=True))
    print(f"\nWrote hardware profile to {Path(args.out).resolve()}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    compressors = _split_csv(args.compressors)
    unknown = [name for name in compressors if name not in compressor_names()]
    if unknown:
        raise SystemExit(f"Unknown compressors: {', '.join(unknown)}")
    runtime = get_runtime(args.runtime)
    if not runtime.is_available():
        print(f"Warning: runtime {args.runtime!r} did not report available; benchmark will surface runtime errors.")
    runtime_options = json.loads(args.runtime_options) if args.runtime_options else {}
    config = BenchmarkConfig(
        runtime=args.runtime,
        model=args.model,
        dataset=args.dataset,
        compressors=compressors,
        out=args.out,
        max_tasks=args.max_tasks,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        timeout_s=args.timeout_s,
        runtime_options=runtime_options,
    )
    results = BenchmarkRunner(runtime).run(config)
    print(f"Wrote {len(results)} result rows to {Path(args.out).resolve() / 'results.jsonl'}")
    print(f"Report: {Path(args.out).resolve() / 'report.md'}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    report, failures = generate_report(args.run_dir)
    print(f"Report: {report.resolve()}")
    print(f"Failures: {failures.resolve()}")
    return 0


def cmd_single(args: argparse.Namespace) -> int:
    runtime = get_runtime(args.runtime)
    document_path = Path(args.document)
    document = document_path.read_text(encoding="utf-8", errors="replace")
    out_dir = Path(args.out or f"runs/single_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "single_task.jsonl"
    task = BenchmarkTask(
        id="single_001",
        type="single_user_document",
        document=document,
        question=args.question,
        gold_answer="",
        metadata={"source_path": str(document_path)},
    )
    write_tasks(dataset_path, [task])
    compressors = ["raw"] if args.compressor == "raw" else ["raw", args.compressor]
    config = BenchmarkConfig(
        runtime=args.runtime,
        model=args.model,
        dataset=str(dataset_path),
        compressors=compressors,
        out=str(out_dir),
        max_tasks=1,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    results = BenchmarkRunner(runtime).run(config)
    print(f"Single-document run: {out_dir.resolve()}")
    for record in results:
        print(
            f"\n[{record['compressor']}] prompt_tokens={record['compressed_prompt_tokens']} "
            f"time_s={record['compressed_total_time_s']:.4f}"
        )
        print(record["answer"])
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    budgets = [float(value) for value in _split_csv(args.budgets)]
    root = Path(args.out or f"runs/ablate_{args.compressor}_{int(time.time())}")
    root.mkdir(parents=True, exist_ok=True)
    runtime = get_runtime(args.runtime)
    all_rows = []
    for budget in budgets:
        subdir = root / f"budget_{str(budget).replace('.', '_')}"
        config = BenchmarkConfig(
            runtime=args.runtime,
            model=args.model,
            dataset=args.dataset,
            compressors=[args.compressor],
            out=str(subdir),
            max_tasks=args.max_tasks,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            budget=budget,
            metadata={"ablation_budget": budget},
        )
        rows = BenchmarkRunner(runtime).run(config)
        for row in rows:
            row["ablation_budget"] = budget
        all_rows.extend(rows)
    from sclab.utils.jsonl import write_jsonl

    write_jsonl(root / "results.jsonl", all_rows)
    generate_report(root)
    print(f"Ablation report: {(root / 'report.md').resolve()}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    tasks = ingest_documents(args.path, args.out)
    print(f"Ingested {len(tasks)} documents into {Path(args.out).resolve()}")
    return 0


def cmd_compress(args: argparse.Namespace) -> int:
    import sys

    text = Path(args.document).read_text(encoding="utf-8", errors="replace")
    compressor = get_compressor(args.compressor, budget=args.budget)
    result = compressor.compress(Document(text=text), args.question)
    print(result.compressed_text)
    if args.stats:
        print(
            f"[{compressor.name}] {result.original_tokens} -> {result.compressed_tokens} tokens "
            f"({result.compression_ratio_tokens:.0%})",
            file=sys.stderr,
        )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from sclab.server import serve as run_server

    backend = args.backend
    model = args.model
    upstream = args.upstream
    # Smart defaults so `sclab up` just works:
    if backend == "auto" and not upstream and not model:
        # No model given -> if an Ollama model is running, proxy it; else Orthrus 4B.
        from sclab.proxy import discover_ollama_model

        found = discover_ollama_model()
        if found:
            backend, upstream, model = "proxy", "http://localhost:11434/v1", found
            print(f"Auto-detected Ollama model '{found}' — proxying it (model-agnostic mode).")
        else:
            model = "orthrus-qwen3-4b"
    elif not model:
        model = "orthrus-qwen3-4b"

    run_server(
        model=model, host=args.host, port=args.port,
        mode=args.mode, block_size=args.block_size, backoff=args.backoff,
        served_name=args.served_name, backend=backend, upstream=upstream,
        api_key=args.api_key, open_dashboard=getattr(args, "open_dash", False) or getattr(args, "_is_up", False),
    )
    return 0


def cmd_hermes_config(args: argparse.Namespace) -> int:
    base = f"http://{args.host}:{args.port}/v1"
    print("Add a Custom OpenAI-compatible provider in Hermes desktop with:\n")
    print(f"  Base URL:  {base}")
    print(f"  API key:   local            (any value — it is not checked)")
    print(f"  Model:     {args.model}\n")
    print("Config file (if you edit it directly): ~/.hermes/config.yaml\n")
    print("providers:")
    print("  matryoshka:")
    print("    type: openai")
    print(f"    base_url: {base}")
    print("    api_key: local")
    print(f"    model: {args.model}")
    print(f"\nDashboard while you chat:  http://{args.host}:{args.port}/dashboard")
    return 0


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def compare_raw_and_compressed(
    runtime_name: str,
    model: str,
    document: str,
    question: str,
    compressor_name: str,
) -> dict[str, object]:
    runtime = get_runtime(runtime_name)
    compressor = get_compressor(compressor_name)
    doc = Document(text=document)
    compressed = compressor.compress(doc, question)
    request = GenerationRequest(
        model=model,
        prompt=f"Source:\n{compressed.compressed_text}\n\nQuestion:\n{question}\n\nAnswer:\n",
    )
    result = runtime.generate(request)
    return {
        "compressor": compressor.name,
        "tokens": count_tokens(request.prompt, model=model).to_dict(),
        "result": result.to_dict(),
    }
