<# Start the separate GM handbook assistant; the existing NPC task is untouched. #>
param([switch]$BuildIndex)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $PythonExe)) { throw 'Run ps\Install-NpcRag.ps1 first.' }
Set-Location -LiteralPath $ProjectRoot
if ($BuildIndex) {
    & $PythonExe -m npc_rag gm-build-index # Rebuild from GM_WIKI_SOURCE before launching.
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $PythonExe -m npc_rag gm-serve # Foreground process for a console or scheduled task.
exit $LASTEXITCODE
