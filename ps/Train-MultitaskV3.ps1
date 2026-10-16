<# Run the targeted v3 experiment, export it, and compare with v2 without changing live configuration. #>
param([string]$RunName = 'multitask-v3-hired-opening-20261015')
$ErrorActionPreference = 'Stop'
$V3Root = Split-Path -Parent $PSScriptRoot
$V3Python = Join-Path $V3Root '.venv-train\Scripts\python.exe'
$V3Run = Join-Path $V3Root ('runs\' + $RunName)
$V3Release = Join-Path $V3Root 'runs\releases\decider-multitask-v3-q4km'
$V3Status = Join-Path $V3Root ('runs\' + $RunName + '-pipeline.json')
if ((Test-Path -LiteralPath $V3Run) -or (Test-Path -LiteralPath $V3Release)) { throw 'Choose a fresh run/release; existing artifacts are preserved.' }
function Write-V3Status([string]$Stage, [string]$Message = '') {
    @{stage=$Stage; message=$Message; run=$V3Run; release=$V3Release; deployed=$false; pid=$PID; updated_utc=[DateTime]::UtcNow.ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath $V3Status -Encoding UTF8
    Write-Output $Stage # A separate pipeline status exists before the trainer creates its new run directory.
}
Push-Location $V3Root
try {
    Write-V3Status 'training'
    & $V3Python python\train_classifier.py --task multitask --local-files-only --epochs 2 --participation-dataset data\participation\synthetic-expanded-v3.jsonl --participation-dev-dataset data\participation\development-v3.jsonl --output $V3Run
    if ($LASTEXITCODE -ne 0) { throw 'V3 training failed.' }
    Write-V3Status 'export_and_same_data_comparison'
    & (Join-Path $PSScriptRoot 'Finish-MultitaskDecider.ps1') -Run $V3Run -ExpandedParticipation -Name decider-multitask-v3 -Release $V3Release -BaselineRelease (Join-Path $V3Root 'runs\releases\decider-multitask-v2-q4km') -BaselineGgufFile decider-multitask-v2.gguf
    Write-V3Status 'targeted_replay'
    & $V3Python python\report_multitask_v3.py --run $V3Run --release $V3Release --baseline-release runs\releases\decider-multitask-v2-q4km --replay data\participation\seeded-opening-regression-v3.jsonl
    if ($LASTEXITCODE -ne 0) { throw 'V3 targeted replay failed.' }
    Write-V3Status 'completed'
} catch {
    Write-V3Status 'failed' $_.Exception.Message
    throw
} finally {
    Pop-Location
}
