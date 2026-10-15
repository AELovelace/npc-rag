<# Train one experimental adapter for GAME/CHAT and JOIN/QUIET using the cached base. #>
param(
    [switch]$SmokeTest,
    [switch]$CheckData,
    [int]$Epochs = 2,
    [string]$Output = ""
)
$ErrorActionPreference = 'Stop'
$MultitaskRoot = Split-Path -Parent $PSScriptRoot
$MultitaskPython = Join-Path $MultitaskRoot '.venv-train\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $MultitaskPython)) { throw 'Run ps\Install-ClassifierTraining.ps1 first.' }
$MultitaskArgs = @((Join-Path $MultitaskRoot 'python\train_classifier.py'), '--task', 'multitask', '--local-files-only', '--epochs', [string]$Epochs)
if ($SmokeTest) { $MultitaskArgs += '--smoke-test' }
if ($CheckData) { $MultitaskArgs += '--check-data' }
if ($Output) { $MultitaskArgs += @('--output', $Output) }
& $MultitaskPython @MultitaskArgs # Keep each argument separate so paths containing spaces are safe.
if ($LASTEXITCODE -ne 0) { throw 'Multi-task training/check failed; see the error above.' }
