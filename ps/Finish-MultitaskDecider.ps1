<# Finish a local experiment: compare specialists, merge, export, then evaluate one GGUF on both tasks. #>
param(
    [Parameter(Mandatory = $true)][string]$Run,
    [int]$WaitForProcessId = 0,
    [string]$Name = 'decider-multitask-v1',
    [string]$Release = ''
)
$ErrorActionPreference = 'Stop'
$ExperimentRoot = Split-Path -Parent $PSScriptRoot
$ExperimentPython = Join-Path $ExperimentRoot '.venv-train\Scripts\python.exe'
if ([IO.Path]::IsPathRooted($Run)) {
    $ExperimentRun = [IO.Path]::GetFullPath($Run)
} else {
    $ExperimentRun = [IO.Path]::GetFullPath((Join-Path $ExperimentRoot $Run))
}
if (-not $Release) { $Release = Join-Path $ExperimentRoot ('runs\releases\' + $Name + '-q4km') }
$ExperimentStatusPath = Join-Path $ExperimentRun 'experiment-status.json'

function Write-ExperimentStatus([string]$Stage, [string]$ErrorMessage = '') {
    @{ stage = $Stage; updated_utc = [DateTime]::UtcNow.ToString('o'); error = $ErrorMessage;
       run = $ExperimentRun; release = $Release; deployment_ready = $false } |
        ConvertTo-Json | Set-Content -LiteralPath $ExperimentStatusPath -Encoding UTF8
    Write-Output $Stage # Persist the active stage so the background experiment remains inspectable.
}

Push-Location $ExperimentRoot
try {
    Write-ExperimentStatus 'waiting_for_training'
    if ($WaitForProcessId -gt 0) {
        $ExperimentProcess = Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue
        if ($ExperimentProcess) { $ExperimentProcess | Wait-Process -Timeout 14400 }
    }
    $ExperimentCompletion = Get-Content -LiteralPath (Join-Path $ExperimentRun 'completion.json') -Raw | ConvertFrom-Json
    if (-not $ExperimentCompletion.completed -or $ExperimentCompletion.smoke_test) { throw 'A completed full training run is required.' }
    Write-ExperimentStatus 'comparing_specialists'
    & $ExperimentPython python\compare_multitask_runs.py --run $ExperimentRun
    if ($LASTEXITCODE -ne 0) { throw 'Specialist comparison failed.' }
    Write-ExperimentStatus 'merging_adapter'
    & $ExperimentPython python\merge_classifier.py --task multitask --run $ExperimentRun --local-files-only
    if ($LASTEXITCODE -ne 0) { throw 'Adapter merge failed.' }
    Write-ExperimentStatus 'exporting_gguf'
    & $ExperimentPython python\export_classifier_gguf.py --task multitask --run $ExperimentRun --name $Name --output $Release
    if ($LASTEXITCODE -ne 0) { throw 'GGUF export failed.' }
    Write-ExperimentStatus 'evaluating_gguf'
    & $ExperimentPython python\evaluate_multitask.py --release $Release --gguf-file ($Name + '.gguf')
    if ($LASTEXITCODE -ne 0) { throw 'GGUF evaluation failed.' }
    Write-ExperimentStatus 'completed'
} catch {
    Write-ExperimentStatus 'failed' $_.Exception.Message
    throw # Keep failed stages visible; never mark a partial experiment completed.
} finally {
    Pop-Location
}
