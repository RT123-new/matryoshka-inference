from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _run(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=3, check=False)
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _sysctl(name: str) -> str | None:
    return _run(["sysctl", "-n", name])


def scan_hardware() -> dict[str, Any]:
    system = platform.system()
    profile: dict[str, Any] = {
        "platform": {
            "system": system,
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "runtimes": {
            "ollama": shutil.which("ollama"),
            "mlx_lm_generate": shutil.which("mlx_lm.generate"),
            "llama_cli": shutil.which("llama-cli"),
            "nvidia_smi": shutil.which("nvidia-smi"),
        },
        "memory": {},
        "gpu": {},
        "notes": [],
    }

    try:
        import psutil

        memory = psutil.virtual_memory()
        profile["memory"].update(
            {
                "total_bytes": memory.total,
                "available_bytes": memory.available,
                "percent_used": memory.percent,
            }
        )
    except Exception as exc:
        profile["notes"].append(f"psutil memory detection failed: {exc}")

    if system == "Darwin":
        profile["platform"]["chip_name"] = _sysctl("machdep.cpu.brand_string") or _sysctl("hw.model")
        memsize = _sysctl("hw.memsize")
        if memsize and memsize.isdigit():
            profile["memory"]["sysctl_total_bytes"] = int(memsize)
        pressure = _run(["memory_pressure"])
        if pressure:
            profile["memory"]["memory_pressure"] = pressure.splitlines()[-10:]
        profile["gpu"]["metal_likely_available"] = platform.machine() in {"arm64", "x86_64"}
        profile["gpu"]["apple_silicon"] = platform.machine() == "arm64"
    elif system in {"Linux", "Windows"}:
        nvidia = shutil.which("nvidia-smi")
        if nvidia:
            profile["gpu"]["nvidia_smi"] = _run([nvidia, "--query-gpu=name,memory.total", "--format=csv,noheader"])

    return profile


def write_hardware_profile(path: str | Path) -> dict[str, Any]:
    profile = scan_hardware()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    return profile
