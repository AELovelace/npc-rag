<# Train the separate sociable JOIN/QUIET pilot using the existing CUDA training environment. #>
param(
    [switch]$SmokeTest,
    [switch]$CheckData,
    [switch]$LocalFilesOnly,
    [int]$Epochs = 2,
    [string]$Output = ""
)
$ErrorActionPreference = 'Stop'
$ParticipationRoot = Split-Path -Parent $PSScriptRoot
$ParticipationPython = Join-Path $ParticipationRoot '.venv-train\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ParticipationPython)) {
    throw 'Run ps\Install-ClassifierTraining.ps1 first.'
}
$ParticipationArgs = @((Join-Path $ParticipationRoot 'python\train_classifier.py'), '--task', 'participation', '--epochs', [string]$Epochs)
if ($SmokeTest) { $ParticipationArgs += '--smoke-test' }
if ($CheckData) { $ParticipationArgs += '--check-data' }
if ($LocalFilesOnly) { $ParticipationArgs += '--local-files-only' }
if ($Output) { $ParticipationArgs += @('--output', $Output) }
& $ParticipationPython @ParticipationArgs # Arguments stay separate so paths containing spaces are safe.
if ($LASTEXITCODE -ne 0) { throw 'Participation training/check failed; see the error above.' }
