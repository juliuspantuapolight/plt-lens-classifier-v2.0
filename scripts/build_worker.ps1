# Freeze src/worker.py into desktop/worker-dist/tray-worker/ (no Python or torch needed on the target PC).
# Builds in .venv-build, which has only the inspection runtime, so torch can never be bundled by accident.
# Requires model.onnx in each model folder: uv run python scripts/export_onnx.py models/tlens-mobilenet-bare models/tlens-mobilenet
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

foreach ($dir in 'models/tlens-mobilenet-bare', 'models/tlens-mobilenet') {
    if (-not (Test-Path "$dir/model.onnx")) { throw "$dir/model.onnx is missing; run scripts/export_onnx.py first." }
}

$env:UV_PROJECT_ENVIRONMENT = '.venv-build'
uv sync --no-default-groups --group build
if ($LASTEXITCODE) { throw 'uv sync failed' }
uv run --no-sync pyinstaller --noconfirm --onedir --name tray-worker --paths . `
    --exclude-module torch --exclude-module transformers `
    --distpath desktop/worker-dist --workpath build/pyinstaller --specpath build/pyinstaller src/worker.py
if ($LASTEXITCODE) { throw 'PyInstaller failed' }
Write-Host 'Built desktop/worker-dist/tray-worker/tray-worker.exe'
