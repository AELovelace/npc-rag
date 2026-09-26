<#
.SYNOPSIS
    Start npc-rag (port 9092) in the background and wait until it is healthy.
.PARAMETER Foreground
    Run it in this window instead (used by the scheduled task).
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File ps\Start-NpcRag.ps1
#>
param(
    [switch]$Foreground
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv\Scripts\python.exe"
$Logs = Join-Path $Root "logs"
if (-not (Test-Path $Venv)) { throw "Run ps\Install-NpcRag.ps1 first." }
New-Item -ItemType Directory -Force $Logs | Out-Null

if ($Foreground) {
    Set-Location $Root
    & $Venv -m npc_rag serve    # Blocks until the service stops.
    exit $LASTEXITCODE
}

# Read NPC_PORT from the environment or .env so the health check hits the right port.
$port = [Environment]::GetEnvironmentVariable("NPC_PORT")
$envFile = Join-Path $Root ".env"
if (-not $port -and (Test-Path $envFile)) {
    $line = Select-String -Path $envFile -Pattern "^\s*NPC_PORT\s*=\s*(\d+)" | Select-Object -First 1
    if ($line) { $port = $line.Matches[0].Groups[1].Value }
}
$port = if ($port) { [int]$port } else { 9092 }

$busy = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($busy) { Write-Host "Port $port already in use (PID $($busy[0].OwningProcess)); not starting." -ForegroundColor Yellow; return }

Start-Process -FilePath $Venv -ArgumentList "-m", "npc_rag", "serve" -WorkingDirectory $Root `
    -WindowStyle Hidden -RedirectStandardOutput "$Logs\npc-rag.out.log" -RedirectStandardError "$Logs\npc-rag.log"

$deadline = (Get-Date).AddSeconds(120)
while ((Get-Date) -lt $deadline) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 5
        Write-Host "npc-rag is up on port $port" -ForegroundColor Green
        if (-not $health.llm) {
            Write-Host "Warning: no reply model at $($health.llm_url). Replies will use fallback lines." -ForegroundColor Yellow
        }
        if (-not $health.classifier) {
            Write-Host "Warning: no classifier at $($health.classifier_url). The wiki score will decide game vs chat." -ForegroundColor Yellow
        }
        return
    } catch { Start-Sleep -Seconds 1 }  # Model loading takes a few seconds on first start.
}
throw "npc-rag did not become healthy on port $port; see $Logs\npc-rag.log"
