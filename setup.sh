#!/usr/bin/env bash
# One-time setup (Linux): installs uv, Python 3.12 venv, CUDA torch, NeMo Speech.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

[ -d .venv ] || uv venv --python 3.12 .venv
PY=.venv/bin/python

echo "Installing PyTorch (CUDA wheels)..."
uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu128 torch

if [ ! -f .deps/nemo-speech/pyproject.toml ]; then
  echo "Fetching NeMo Speech (main branch, as required by the Nemotron-3 ASR guide)..."
  mkdir -p .deps
  curl -sL https://github.com/NVIDIA-NeMo/Speech/archive/refs/heads/main.tar.gz \
    | tar xz -C .deps
  mv .deps/Speech-main .deps/nemo-speech 2>/dev/null || true
fi

echo "Installing NeMo [asr] + app dependencies..."
uv pip install --python "$PY" Cython packaging
uv pip install --python "$PY" -e '.deps/nemo-speech[asr]'
uv pip install --python "$PY" imageio-ffmpeg soundfile fastapi 'uvicorn[standard]' python-multipart

echo
echo "Done. Start the app with:  ./run.sh"
