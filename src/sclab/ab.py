"""A/B comparison: baseline vs Matryoshka on the SAME model + prompts.

For a stock Ollama model (like whatever Hermes is currently using), "Matryoshka
on" means the semantic-compression layer: the prompt is compressed before it is
sent, so the model prefills fewer tokens and returns the answer sooner. The
diffusion acceleration path only applies to Orthrus MLX checkpoints, so it is
deliberately not part of this comparison — this measures the lever that is real
for the current model.

Every prompt is run twice with identical settings and identical measurements:
prompt tokens, time-to-first-token, total wall time, decode tok/s, and
end-to-end tok/s (completion / total time — the number that actually improves
when prefill shrinks). Answer quality is scored against a gold answer so a speed
win that loses accuracy is visible.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Optional

from sclab.benchmarks.datasets import load_tasks
from sclab.benchmarks.runner import render_prompt
from sclab.benchmarks.scoring import score_answer
from sclab.compressors import Document, get_compressor
from sclab.runtimes.base import GenerationRequest
from sclab.runtimes.ollama import OllamaRuntime


def read_hermes_model(config_path: str = "~/.hermes/config.yaml") -> tuple[Optional[str], Optional[str], Optional[int]]:
    """Return (model, ollama_base_url, num_ctx) from Hermes' config, if present."""
    p = Path(config_path).expanduser()
    if not p.exists():
        return None, None, None
    text = p.read_text()
    model = (re.search(r"default:\s*(\S+)", text) or [None, None])[1]
    base = (re.search(r"base_url:\s*(\S+)", text) or [None, None])[1]
    ctx_m = re.search(r"num_ctx:\s*(\d+)", text)
    num_ctx = int(ctx_m.group(1)) if ctx_m else None
    # OllamaRuntime hits the native API; strip the OpenAI /v1 suffix.
    if base:
        base = base.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
    return model, base, num_ctx


def _eff_tps(completion: Optional[int], total_s: Optional[float]) -> Optional[float]:
    if completion and total_s and total_s > 0:
        return completion / total_s
    return None


def run_ab(
    model: str,
    dataset: str,
    out_dir: str,
    base_url: str = "http://127.0.0.1:11434",
    compressor_name: str = "extractive_relevance",
    max_tokens: int = 220,
    max_tasks: Optional[int] = None,
    num_ctx: Optional[int] = None,
    progress=lambda s: None,
) -> dict[str, Any]:
    rt = OllamaRuntime(base_url)
    compressor = get_compressor(compressor_name)
    tasks = load_tasks(dataset, max_tasks)
    opts = {"think": False}
    if num_ctx:
        opts["options"] = {"num_ctx": num_ctx}

    def gen(prompt: str):
        return rt.generate(GenerationRequest(
            model=model, prompt=prompt, max_tokens=max_tokens,
            temperature=0.0, runtime_options=opts,
        ))

    # Warm up: load the model into memory so the first timed request does not
    # eat the cold-load penalty (which would fake a huge speedup on task 1).
    progress("warming up model (loading weights)...")
    rt.generate(GenerationRequest(model=model, prompt="Reply with: ready",
                                  max_tokens=8, temperature=0.0, runtime_options=opts))

    rows: list[dict[str, Any]] = []
    for i, t in enumerate(tasks, 1):
        progress(f"[{i}/{len(tasks)}] {t.id} ({t.type})")
        raw_prompt = render_prompt("raw", t.document, t.question)
        comp = compressor.compress(Document(text=t.document), t.question)
        comp_prompt = render_prompt(compressor.name, comp.compressed_text, t.question)

        base = gen(raw_prompt)
        matr = gen(comp_prompt)
        base_q = score_answer(base.text, t).quality_score if base.text else 0.0
        matr_q = score_answer(matr.text, t).quality_score if matr.text else 0.0

        b_total, m_total = base.total_time_s, matr.total_time_s
        rows.append({
            "id": t.id, "area": t.type, "question": t.question,
            "gold_answer": t.gold_answer,
            "baseline": {
                "prompt_tokens": base.prompt_tokens, "completion_tokens": base.completion_tokens,
                "ttft_s": base.time_to_first_token_s, "total_s": b_total,
                "decode_tps": base.decode_tokens_per_s, "eff_tps": _eff_tps(base.completion_tokens, b_total),
                "quality": round(base_q, 3), "output": base.text,
            },
            "matryoshka": {
                "prompt_tokens": matr.prompt_tokens, "completion_tokens": matr.completion_tokens,
                "ttft_s": matr.time_to_first_token_s, "total_s": m_total,
                "decode_tps": matr.decode_tokens_per_s, "eff_tps": _eff_tps(matr.completion_tokens, m_total),
                "quality": round(matr_q, 3), "output": matr.text,
                "compressed_prompt": comp_prompt,
            },
            "compression_ratio": comp.compression_ratio_tokens,
            "speedup_total": round(b_total / m_total, 2) if (b_total and m_total) else None,
        })

    meta = {"model": model, "base_url": base_url, "compressor": compressor.name,
            "max_tokens": max_tokens, "num_ctx": num_ctx, "tasks": len(rows)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (out / "report.html").write_text(build_html(meta, rows), encoding="utf-8")
    (out / "report.md").write_text(build_md(meta, rows), encoding="utf-8")
    return {"meta": meta, "rows": rows, "out": str(out)}


def _avg(vals: list[Optional[float]]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def build_md(meta: dict, rows: list[dict]) -> str:
    sp = _avg([r["speedup_total"] for r in rows])
    cr = _avg([r["compression_ratio"] for r in rows])
    bq = _avg([r["baseline"]["quality"] for r in rows])
    mq = _avg([r["matryoshka"]["quality"] for r in rows])
    lines = [
        f"# Matryoshka A/B — {meta['model']}",
        "",
        f"Same model, same prompts, compressor = `{meta['compressor']}`. "
        f"Baseline = raw prompt; Matryoshka = compressed prompt.",
        "",
        f"- Avg end-to-end speedup: **{sp:.2f}x**" if sp else "- speedup: n/a",
        f"- Avg prompt size after compression: **{cr*100:.0f}%** of raw" if cr else "",
        f"- Avg quality: baseline {bq:.3f} vs Matryoshka {mq:.3f}" if (bq is not None) else "",
        "",
        "| area | prompt tok (base→matr) | total s (base→matr) | eff tok/s (base→matr) | speedup | quality |",
        "|---|---|---|---|---:|---|",
    ]
    for r in rows:
        b, m = r["baseline"], r["matryoshka"]
        lines.append(
            f"| {r['area']} | {b['prompt_tokens']}→{m['prompt_tokens']} | "
            f"{_fmt(b['total_s'])}→{_fmt(m['total_s'])} | "
            f"{_fmt(b['eff_tps'])}→{_fmt(m['eff_tps'])} | "
            f"{r['speedup_total'] or '—'}x | {b['quality']}→{m['quality']} |"
        )
    return "\n".join(l for l in lines if l is not None)


def _fmt(v) -> str:
    return f"{v:.1f}" if isinstance(v, (int, float)) else "—"


def build_html(meta: dict, rows: list[dict]) -> str:
    sp = _avg([r["speedup_total"] for r in rows]) or 0
    cr = _avg([r["compression_ratio"] for r in rows]) or 1
    bq = _avg([r["baseline"]["quality"] for r in rows]) or 0
    mq = _avg([r["matryoshka"]["quality"] for r in rows]) or 0

    def esc(s):
        return html.escape(str(s or ""))

    cards = []
    for r in rows:
        b, m = r["baseline"], r["matryoshka"]
        cards.append(f"""
        <div class="card">
          <div class="qhead"><span class="chip">{esc(r['area'])}</span> {esc(r['question'])}</div>
          <div class="gold">Gold answer: {esc(r['gold_answer'])}</div>
          <div class="cols">
            <div class="col base">
              <div class="ctitle">Baseline (no Matryoshka)</div>
              <div class="metrics">
                <span>prompt <b>{esc(b['prompt_tokens'])}</b> tok</span>
                <span>total <b>{_fmt(b['total_s'])}</b> s</span>
                <span>decode <b>{_fmt(b['decode_tps'])}</b> tok/s</span>
                <span>end-to-end <b>{_fmt(b['eff_tps'])}</b> tok/s</span>
                <span class="q">quality <b>{b['quality']}</b></span>
              </div>
              <div class="out">{esc(b['output'])}</div>
            </div>
            <div class="col matr">
              <div class="ctitle">Matryoshka ON (compressed prompt)</div>
              <div class="metrics">
                <span>prompt <b>{esc(m['prompt_tokens'])}</b> tok</span>
                <span>total <b>{_fmt(m['total_s'])}</b> s</span>
                <span>decode <b>{_fmt(m['decode_tps'])}</b> tok/s</span>
                <span>end-to-end <b>{_fmt(m['eff_tps'])}</b> tok/s</span>
                <span class="q">quality <b>{m['quality']}</b></span>
              </div>
              <div class="out">{esc(m['output'])}</div>
            </div>
          </div>
          <div class="cmp">
            prompt {esc(b['prompt_tokens'])} → {esc(m['prompt_tokens'])} tok
            ({(r['compression_ratio'] or 1)*100:.0f}% of raw) ·
            <b>{r['speedup_total'] or '—'}× faster</b> end-to-end ·
            quality {b['quality']} → {m['quality']}
          </div>
        </div>""")

    return f"""<!doctype html><html><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Matryoshka A/B — {esc(meta['model'])}</title>
<style>
 body{{margin:0;background:#0a0e14;color:#e6edf3;font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
 .wrap{{max-width:1100px;margin:0 auto;padding:22px}}
 h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#7d8896;font-size:13px;margin-bottom:18px}}
 .kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:22px}}
 .kpi{{background:#121821;border:1px solid #1e2733;border-radius:12px;padding:14px 16px}}
 .kpi .l{{color:#7d8896;font-size:11px;text-transform:uppercase;letter-spacing:.5px}}
 .kpi .v{{font-size:26px;font-weight:680;margin-top:3px;color:#2dd4bf}}
 .card{{background:#121821;border:1px solid #1e2733;border-radius:12px;padding:16px;margin-bottom:16px}}
 .qhead{{font-weight:600;margin-bottom:4px}} .gold{{color:#7d8896;font-size:12px;margin-bottom:12px}}
 .chip{{background:rgba(45,212,191,.14);color:#2dd4bf;font-size:10.5px;padding:2px 8px;border-radius:999px;margin-right:8px;text-transform:uppercase;letter-spacing:.4px}}
 .cols{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
 @media(max-width:760px){{.cols{{grid-template-columns:1fr}}.kpis{{grid-template-columns:repeat(2,1fr)}}}}
 .col{{border:1px solid #1e2733;border-radius:10px;padding:12px;background:#0d131b}}
 .col.matr{{border-color:rgba(45,212,191,.35)}}
 .ctitle{{font-size:12px;font-weight:600;margin-bottom:8px}} .col.base .ctitle{{color:#f5a623}} .col.matr .ctitle{{color:#2dd4bf}}
 .metrics{{display:flex;flex-wrap:wrap;gap:10px;font-size:11.5px;color:#7d8896;margin-bottom:10px}}
 .metrics b{{color:#e6edf3;font-family:ui-monospace,Menlo,monospace}}
 .out{{white-space:pre-wrap;font-size:12.5px;background:#0a0e14;border:1px solid #1e2733;border-radius:8px;padding:10px;max-height:220px;overflow:auto}}
 .cmp{{margin-top:12px;font-size:12.5px;color:#7d8896;border-top:1px solid #1e2733;padding-top:10px}} .cmp b{{color:#2dd4bf}}
</style></head><body><div class="wrap">
 <h1>Matryoshka A/B comparison</h1>
 <div class="sub">Model <b>{esc(meta['model'])}</b> · endpoint {esc(meta['base_url'])} · compressor <b>{esc(meta['compressor'])}</b> · {meta['tasks']} prompts · same model &amp; settings both sides. Baseline = raw prompt, Matryoshka = compressed prompt.</div>
 <div class="kpis">
   <div class="kpi"><div class="l">Avg speedup (end-to-end)</div><div class="v">{sp:.2f}×</div></div>
   <div class="kpi"><div class="l">Avg prompt size</div><div class="v">{cr*100:.0f}%</div></div>
   <div class="kpi"><div class="l">Quality baseline</div><div class="v" style="color:#f5a623">{bq:.2f}</div></div>
   <div class="kpi"><div class="l">Quality Matryoshka</div><div class="v">{mq:.2f}</div></div>
 </div>
 {''.join(cards)}
 <div class="sub" style="margin-top:8px">Note: this model is a stock Ollama GGUF, so "Matryoshka on" = semantic prompt compression (less prefill). Diffusion acceleration applies only to Orthrus MLX checkpoints. Decode tok/s is roughly unchanged by compression; the win shows in prompt size, time-to-first-token and total time — captured here as end-to-end tok/s and speedup.</div>
</div></body></html>"""
