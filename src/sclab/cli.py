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
    from sclab import __version__

    parser = argparse.ArgumentParser(prog="sclab", description="Semantic Compression Lab CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
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

    ab = subparsers.add_parser(
        "ab", help="A/B: baseline vs Matryoshka on the same model + prompts (HTML report)")
    ab.add_argument("--dataset", default="data/tasks/diverse_ab.jsonl")
    ab.add_argument("--model", default=None, help="Defaults to the current Hermes model")
    ab.add_argument("--backend", default="auto", choices=["auto", "ollama", "orthrus"],
                    help="ollama = compression A/B; orthrus = AR-vs-diffusion acceleration A/B")
    ab.add_argument("--base-url", default=None, help="Ollama base URL (defaults to Hermes' setting)")
    ab.add_argument("--block-size", type=int, default=16, help="Diffusion block size (orthrus backend)")
    ab.add_argument("--compressor", default="extractive_relevance")
    ab.add_argument("--max-tokens", type=int, default=220)
    ab.add_argument("--max-tasks", type=int, default=None)
    ab.add_argument("--num-ctx", type=int, default=None, help="Override context size (default: Hermes' setting)")
    ab.add_argument("--out", default=None)
    ab.add_argument("--open", dest="open_report", action="store_true", help="Open the HTML report when done")
    ab.set_defaults(func=cmd_ab)

    hconn = subparsers.add_parser(
        "hermes-connect",
        help="Transparently route Hermes through the dashboard proxy (reversible)")
    hconn.add_argument("--port", type=int, default=8977, help="Port the proxy will listen on")
    hconn.add_argument("--config", default="~/.hermes/config.yaml")
    hconn.add_argument("--revert", action="store_true", help="Restore Hermes' original config")
    hconn.set_defaults(func=cmd_hermes_connect)

    spec = subparsers.add_parser(
        "spec-bench",
        help="API-level verified speculation: probe an OpenAI-compatible "
             "/v1/completions engine, then baseline vs spec if it qualifies (experimental)")
    spec.add_argument("--upstream", default=None,
                      help="Engine base URL, e.g. http://localhost:8080/v1. Must expose echo + "
                           "prompt logprobs (llama-cpp-python does; native llama-server does not). "
                           "Omit with --sim for a local demo.")
    spec.add_argument("--api-key", default="")
    spec.add_argument("--model", default=None, help="Model name the engine serves")
    spec.add_argument("--prompt", default=None, help="Prompt text (or use --prompt-file)")
    spec.add_argument("--prompt-file", default=None)
    spec.add_argument("--max-tokens", type=int, default=256)
    spec.add_argument("--draft-chars", type=int, default=64,
                      help="Max characters proposed per verify round")
    spec.add_argument("--burst-tokens", type=int, default=16,
                      help="Tokens to generate per plain-decode burst when no draft is available")
    spec.add_argument("--warm-file", default=None,
                      help="Text to pre-load into the lookup memory (e.g. a tool schema or the "
                           "document being quoted) so drafts land from the first token")
    spec.add_argument("--cost-probe", action="store_true",
                      help="Also measure the engine's scoring-vs-decoding physics (breakeven acceptance)")
    spec.add_argument("--sim", action="store_true",
                      help="Run against the built-in deterministic sim engine (no real model needed)")
    spec.add_argument("--sim-decode-ms", type=float, default=20.0,
                      help="Sim only: modeled per-token sequential decode cost (ms)")
    spec.add_argument("--sim-shift", type=int, default=0, choices=(0, 1),
                      help="Sim only: logprob convention to emulate (0=classic/OpenAI, "
                           "1=shifted/llama-cpp-python). The probe should detect either.")
    spec.set_defaults(func=cmd_spec_bench)

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
    try:
        runtime_options = json.loads(args.runtime_options) if args.runtime_options else {}
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--runtime-options is not valid JSON: {exc}") from None
    if not isinstance(runtime_options, dict):
        raise SystemExit("--runtime-options must be a JSON object, e.g. '{\"mode\":\"diffusion\"}'")
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
    document = _read_document(document_path)
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
    try:
        budgets = [float(value) for value in _split_csv(args.budgets)]
    except ValueError:
        raise SystemExit(f"--budgets must be comma-separated numbers, got: {args.budgets!r}") from None
    if not budgets:
        raise SystemExit("--budgets is empty; pass e.g. 0.2,0.4,0.6")
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

    text = _read_document(Path(args.document))
    compressor = get_compressor(args.compressor, budget=args.budget)
    result = compressor.compress(Document(text=text), args.question)
    print(result.compressed_text)
    if args.stats:
        ratio = (
            f"{result.compression_ratio_tokens:.0%}"
            if result.compression_ratio_tokens is not None else "n/a"
        )
        print(
            f"[{compressor.name}] {result.original_tokens} -> {result.compressed_tokens} tokens "
            f"({ratio})",
            file=sys.stderr,
        )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from sclab.proxy import discover_ollama_model
    from sclab.server import serve as run_server

    backend = args.backend
    model = args.model
    upstream = args.upstream
    # Smart defaults so `sclab up` just works:
    if backend == "auto" and not upstream and not model:
        # No model given -> if an Ollama model is running, proxy it; else Orthrus 4B.
        found = discover_ollama_model()
        if found:
            backend, upstream, model = "proxy", "http://localhost:11434/v1", found
            print(f"Auto-detected Ollama model '{found}' — proxying it (model-agnostic mode).")
        else:
            model = "orthrus-qwen3-4b"
    elif backend == "proxy":
        # Explicit proxy: default the upstream to local Ollama, and pick its
        # first model when none was named, so `sclab serve --backend proxy`
        # works with zero extra flags.
        if not upstream:
            upstream = "http://localhost:11434/v1"
            print(f"No --upstream given — defaulting to Ollama at {upstream}")
        if not model:
            from sclab.proxy import list_models

            listed = list_models(upstream, args.api_key) or {}
            ids = [m.get("id") for m in listed.get("data") or [] if m.get("id")]
            model = ids[0] if ids else None
            if not model:
                raise SystemExit(
                    "proxy backend needs a model: pass --model, or start the upstream "
                    "so one can be auto-detected."
                )
            print(f"Auto-detected upstream model '{model}'.")
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
    print("  API key:   local            (any value — it is not checked)")
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


def cmd_ab(args: argparse.Namespace) -> int:
    from sclab.ab import read_hermes_model, run_ab, run_ab_orthrus
    from sclab.runtimes.orthrus_mlx import MODEL_ALIASES

    h_model, h_base, h_ctx = read_hermes_model()
    model = args.model or h_model
    if not model:
        raise SystemExit("No model given and none found in Hermes config. Pass --model.")
    backend = args.backend
    if backend == "auto":
        backend = "orthrus" if (model.lower() in MODEL_ALIASES or "orthrus" in model.lower()) else "ollama"
    out = args.out or f"runs/ab_{model.replace(':', '_').replace('/', '_')}"

    import statistics as st
    if backend == "orthrus":
        print(f"Acceleration A/B on Orthrus model '{model}' (AR vs diffusion, block={args.block_size})")
        result = run_ab_orthrus(
            model=model, dataset=args.dataset, out_dir=out, block_size=args.block_size,
            max_tokens=args.max_tokens, max_tasks=args.max_tasks, progress=lambda s: print("  " + s),
        )
        rows = result["rows"]
        sp = st.mean([r["speedup_decode"] for r in rows if r["speedup_decode"]]) if rows else 0
        label = "decode speedup"
    else:
        base_url = args.base_url or h_base or "http://127.0.0.1:11434"
        num_ctx = args.num_ctx if args.num_ctx is not None else h_ctx
        print(f"Compression A/B on '{model}' via {base_url} (num_ctx={num_ctx}), compressor={args.compressor}")
        result = run_ab(
            model=model, dataset=args.dataset, out_dir=out, base_url=base_url,
            compressor_name=args.compressor, max_tokens=args.max_tokens,
            max_tasks=args.max_tasks, num_ctx=num_ctx, progress=lambda s: print("  " + s),
        )
        rows = result["rows"]
        sp = st.mean([r["speedup_total"] for r in rows if r["speedup_total"]]) if rows else 0
        label = "end-to-end speedup"
    report = Path(out) / "report.html"
    print(f"\nAvg {label}: {sp:.2f}x  ·  report: {report.resolve()}")
    if args.open_report:
        import webbrowser
        webbrowser.open(report.resolve().as_uri())
    return 0


def cmd_hermes_connect(args: argparse.Namespace) -> int:
    """Repoint Hermes' model base_url at the local proxy so every chat turn is
    observed by the dashboard — without changing the model Hermes uses."""
    import re
    from pathlib import Path

    cfg = Path(args.config).expanduser()
    if not cfg.exists():
        raise SystemExit(f"Hermes config not found at {cfg}. Is Hermes desktop installed?")
    bak = cfg.with_name(cfg.name + ".matryoshka-bak")
    proxy_url = f"http://127.0.0.1:{args.port}/v1"

    if args.revert:
        if not bak.exists():
            print("No backup found — nothing to revert.")
            return 0
        cfg.write_text(bak.read_text())
        bak.unlink()
        print("Reverted Hermes config to its original endpoint. Restart Hermes to apply.")
        return 0

    text = cfg.read_text()
    m = re.search(r"base_url:\s*(\S+)", text)
    if not m:
        raise SystemExit("Could not find a model base_url in the Hermes config.")
    current = m.group(1)
    if f":{args.port}/" in current:
        print(f"Hermes already points at the proxy ({current}). Nothing to do.")
        return 0
    if not bak.exists():
        bak.write_text(text)  # preserve the true original
    cfg.write_text(text.replace(current, proxy_url))

    print("Connected Hermes to the Matryoshka dashboard proxy.\n")
    print(f"  Hermes base_url:  {current}  ->  {proxy_url}")
    print(f"  Backup saved to:  {bak}\n")
    print("Next:")
    print("  1. Start the proxy in front of your existing endpoint:")
    print(f"       sclab serve --backend proxy --upstream {current} --port {args.port} --open")
    print("  2. Fully quit and reopen Hermes so it reloads the config.")
    print("  3. Chat as usual — the dashboard lights up on every turn.\n")
    print(f"  Dashboard:  http://127.0.0.1:{args.port}/dashboard")
    print("  Disconnect: sclab hermes-connect --revert   (then restart Hermes)")
    print("\n  Note: keep the proxy running while connected — if it stops, Hermes")
    print("  cannot reach the endpoint until you revert.")
    return 0


def cmd_spec_bench(args: argparse.Namespace) -> int:
    from sclab.spec.bench import format_bench, format_cost_probe, run_bench, run_cost_probe
    from sclab.spec.verify import probe_endpoint

    sim_server = None
    upstream, model = args.upstream, args.model
    if args.sim:
        from sclab.spec.sim import LagLM, SimEngine, start_sim_server

        engine = SimEngine(lm=LagLM(lag=10), overhead_ms=2,
                           prefill_ms_per_token=args.sim_decode_ms / 10.0,
                           decode_ms_per_token=args.sim_decode_ms,
                           logprob_shift=args.sim_shift)
        sim_server, upstream = start_sim_server(engine)
        model = model or "sim-lag-lm"
        print(f"Using built-in sim engine at {upstream} "
              f"(modeled decode {args.sim_decode_ms:.0f} ms/token, logprob_shift={args.sim_shift} "
              f"— SIMULATED, not a real model).")
    if not upstream:
        raise SystemExit("Pass --upstream <engine /v1 base URL>, or --sim for a local demo.")
    if not model:
        raise SystemExit("Pass --model <name the engine serves>.")

    if args.prompt_file:
        prompt = _read_document(Path(args.prompt_file))
    elif args.prompt:
        prompt = args.prompt
    elif args.sim:
        prompt = "the quick brown fox jumps over the lazy dog and then the"
    else:
        raise SystemExit("Pass --prompt or --prompt-file.")

    warm = _read_document(Path(args.warm_file)) if args.warm_file else None
    try:
        # Behavioural probe first: shape compatibility is not enough — the
        # positional alignment must be measured, or verification is unsafe.
        cap = probe_endpoint(upstream, args.api_key, model)
        print(f"capability probe: {cap.status} "
              f"(usable={cap.usable}, shift={cap.shift}) — {cap.detail}")
        if not cap.usable:
            print("This endpoint cannot verify drafts through its public API; "
                  "speculation is disabled and only plain generation is possible.")

        result = run_bench(upstream, args.api_key, model, prompt,
                           max_tokens=args.max_tokens, draft_chars=args.draft_chars,
                           burst_tokens=args.burst_tokens, warm_text=warm, capability=cap)
        print(format_bench(result))
        if result.get("spec_available") and not result.get("identical_output"):
            print("\nWARNING: spec output differed from the plain baseline. This is a "
                  "correctness failure — do not trust any speed number for it.")
        if args.cost_probe and cap.usable and not result.get("error"):
            print("\n" + format_cost_probe(run_cost_probe(upstream, args.api_key, model, prompt)))
    finally:
        if sim_server is not None:
            sim_server.shutdown()
    return 0 if not result.get("error") else 1


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _read_document(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SystemExit(f"Could not read document {path}: {exc}") from None


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
