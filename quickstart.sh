#!/usr/bin/env bash
# One-command setup + launch for Matryoshka Inference.
# Creates a venv, installs the package, and starts the server + dashboard.
#
#   ./quickstart.sh                 # auto: proxy a running Ollama model, else Orthrus-4B
#   ./quickstart.sh --model orthrus-qwen3-4b
#   ./quickstart.sh --backend proxy --upstream http://localhost:11434/v1 --model gemma4:latest
#
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
VENV=".venv"

if [ ! -d "$VENV" ]; then
  echo "==> creating venv ($VENV)"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "==> installing matryoshka-inference (this is a one-time step)"
# [orthrus] pulls MLX + transformers for the accelerated Apple-Silicon path.
# Drop the extra (pip install -e .) if you only want the model-agnostic proxy.
pip install -q -e ".[orthrus]" || pip install -q -e .

echo "==> launching server + dashboard"
exec sclab up --open "$@"
