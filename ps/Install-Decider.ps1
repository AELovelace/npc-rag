<#
.SYNOPSIS
    Install Decider in the existing RAG environment using a short Windows build path.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File ps\Install-Decider.ps1
#>
param(
    [string]$Python = "",     # Defaults to the existing npc-rag virtual environment.
    [string]$BuildTemp = ""   # Override with a short writable folder if this checkout has a long path.
)
$ErrorActionPreference = "Stop"
$RagRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $RagRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "RAG Python was not found. Run ps\Install-NpcRag.ps1 first." }
if (-not $BuildTemp) { $BuildTemp = Join-Path $RagRoot '.t' }
$BuildTemp = [IO.Path]::GetFullPath($BuildTemp)
if ($BuildTemp.Length -gt 24) {
    throw "Build path is too long for llama-cpp-python's nested sources. Pass -BuildTemp with a writable absolute path of at most 24 characters, for example C:\t."
}
New-Item -ItemType Directory -Path $BuildTemp -Force | Out-Null # Keep extraction paths below Windows' legacy limit.
$oldTemp = $env:TEMP
$oldTmp = $env:TMP
$oldTmpDir = $env:TMPDIR
try {
    $env:TEMP = $BuildTemp
    $env:TMP = $BuildTemp
    $env:TMPDIR = $BuildTemp # Python chooses its temp directory when the new pip process starts.
    & $Python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
    if ($LASTEXITCODE -ne 0) { throw "Decider needs Python 3.11+ in this virtual environment." }
    Write-Host "Installing Decider with temporary build directory $BuildTemp"
    & $Python -m pip install --no-cache-dir -r (Join-Path $RagRoot 'requirements-decider.txt')
    if ($LASTEXITCODE -ne 0) { throw "Decider installation failed. A source build also needs Visual Studio C++ Build Tools and CMake; see README.md." }
    & $Python -c "from decider.infer import Decider; import llama_cpp" # Verify native imports without loading or downloading weights.
    if ($LASTEXITCODE -ne 0) { throw "Decider native imports failed; resolve the dependency/DLL error before enabling the backend." }
    Write-Host "Decider dependencies installed. Run the classifier evaluation before enabling the backend." -ForegroundColor Green
} finally {
    $env:TEMP = $oldTemp
    $env:TMP = $oldTmp
    $env:TMPDIR = $oldTmpDir # Restore the caller's environment on success, failure or interruption.
}
