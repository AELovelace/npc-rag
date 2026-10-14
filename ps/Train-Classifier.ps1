<# Runs the experimental single-GPU classifier trainer in its separate environment. #>
param(
    [switch]$SmokeTest,
    [switch]$CheckData,
    [int]$Epochs = 2,
    [string]$Output = ""
)
$ErrorActionPreference = 'Stop'
$TrainingRoot = Split-Path -Parent $PSScriptRoot
$TrainingPython = Join-Path $TrainingRoot '.venv-train\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $TrainingPython)) {
    throw 'Run ps\Install-ClassifierTraining.ps1 first.'
}
$TrainingArgs = @((Join-Path $TrainingRoot 'python\train_classifier.py'), '--epochs', [string]$Epochs)
if ($SmokeTest) { $TrainingArgs += '--smoke-test' }
if ($CheckData) { $TrainingArgs += '--check-data' }
if ($Output) { $TrainingArgs += @('--output', $Output) }
& $TrainingPython @TrainingArgs # Native arguments stay separate; paths with spaces remain intact.
if ($LASTEXITCODE -ne 0) { throw 'Classifier training/check failed; see the error above. Production remains unchanged.' }
