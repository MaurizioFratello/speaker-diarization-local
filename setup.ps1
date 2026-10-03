# One-time setup (Windows, PowerShell): uv + Python 3.12 venv + CUDA torch + NeMo Speech.
# Requires Windows 10+ (ships curl.exe and tar.exe). No git, no compilers needed.

$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

if (-not (Test-Path .venv)) {
    uv venv --python 3.12 .venv
}
$PY = ".venv\Scripts\python.exe"

Write-Host "Installing PyTorch (CUDA 12.8 wheels)..."
uv pip install --python $PY --index-url https://download.pytorch.org/whl/cu128 torch

if (-not (Test-Path .deps\nemo-speech\pyproject.toml)) {
    Write-Host "Fetching NeMo Speech (main branch, as required by the Nemotron-3 ASR guide)..."
    New-Item -ItemType Directory -Force .deps | Out-Null
    curl.exe -sL -o .deps\speech.tar.gz https://github.com/NVIDIA-NeMo/Speech/archive/refs/heads/main.tar.gz
    tar xzf .deps\speech.tar.gz -C .deps
    if (Test-Path .deps\nemo-speech) { Remove-Item -Recurse -Force .deps\nemo-speech }
    Rename-Item .deps\Speech-main nemo-speech
}

Write-Host "Installing NeMo [asr] + app dependencies..."
uv pip install --python $PY Cython packaging
uv pip install --python $PY "-e" ".deps\nemo-speech[asr]"
uv pip install --python $PY imageio-ffmpeg soundfile fastapi "uvicorn[standard]" python-multipart

Write-Host "`nDone. Start the app with:  .\run.bat"
