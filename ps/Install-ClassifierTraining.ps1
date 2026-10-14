<# Creates an isolated CUDA training environment; does not modify the running RAG environment. #>
param([string]$Python = "")
$ErrorActionPreference = "Stop"
$TrainingRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $TrainingRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python)) { throw 'Pass -Python with the full path to Python 3.11 or 3.12.' }
& $Python -c "import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,12) else 1)"
if ($LASTEXITCODE -ne 0) { throw 'Use Python 3.11 or 3.12 for the pinned CUDA wheels.' }
$TrainingPython = Join-Path $TrainingRoot '.venv-train\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $TrainingPython)) {
    & $Python -m venv (Join-Path $TrainingRoot '.venv-train') # Keep training dependencies separate from production.
    if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv-train.' }
}
& $TrainingPython -m pip --disable-pip-version-check --timeout 30 --retries 1 install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Training pip update failed.' }
& $TrainingPython -m pip --disable-pip-version-check --timeout 30 --retries 1 install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
& $TrainingPython -m pip --disable-pip-version-check --timeout 30 --retries 1 install -r (Join-Path $TrainingRoot 'requirements-train.txt')
if ($LASTEXITCODE -ne 0) { throw 'Training dependencies failed to install; see pip output.' }
& $TrainingPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Training dependency consistency check failed.' }
& $TrainingPython -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU unavailable; check the NVIDIA driver'; print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'CUDA validation failed. The first-pass helper targets the NVIDIA 3080.' }
Write-Host 'Training environment ready. Run python/train_classifier.py --check-data, then --smoke-test.'
