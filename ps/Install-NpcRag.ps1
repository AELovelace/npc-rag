<#
.SYNOPSIS
    One-time setup on the AI server: virtual environment, packages, .env, embedding model and wiki index.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File ps\Install-NpcRag.ps1
#>
param(
    [string]$Python = ""  # Path to python.exe (3.10+). Empty = first suitable one on PATH or the py launcher.
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot   # Repository root (this script lives in ps\).
Set-Location $Root

# ── Find Python 3.10+ ─────────────────────────────────────────────────────────
if (-not $Python) {
    foreach ($candidate in @("python", "py -3.12", "py -3.11", "py -3.10")) {
        $parts = $candidate.Split(" ")
        try {
            $ver = & $parts[0] $parts[1..9] -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        } catch { continue }  # Not installed: try the next one.
        if ($ver -and [version]$ver -ge [version]"3.10") {
            $Python = (& $parts[0] $parts[1..9] -c "import sys; print(sys.executable)")
            break
        }
    }
}
if (-not $Python) { throw "Python 3.10 or newer was not found. Install it from python.org, then rerun." }
Write-Host "Using $Python"

# ── Virtual environment + packages ────────────────────────────────────────────
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $Python -m venv .venv                       # Private package folder for this project only.
}
$Venv = Join-Path $Root ".venv\Scripts\python.exe"
& $Venv -m pip install --upgrade pip --quiet
& $Venv -m pip install -r requirements.txt --quiet  # fastembed (CPU/ONNX), fastapi, uvicorn, httpx, numpy, pytest.
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# ── Settings file ─────────────────────────────────────────────────────────────
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example - review LLM_URL and set NPC_API_KEY." -ForegroundColor Yellow
}

# ── Download the embedding model and build the wiki index ─────────────────────
& $Venv -m npc_rag build-index
if ($LASTEXITCODE -ne 0) { throw "Index build failed (is the wiki reachable? check WIKI_SOURCE in .env)" }

# ── Self-test ─────────────────────────────────────────────────────────────────
& $Venv -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
Write-Host "Installed. Start with: powershell -ExecutionPolicy Bypass -File ps\Start-NpcRag.ps1" -ForegroundColor Green
