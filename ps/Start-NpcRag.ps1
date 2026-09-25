<#
.SYNOPSIS
    Start the RAG service (9092) and the classifier/agent service (9091) in the background and wait until both are healthy.
.PARAMETER Service
    all (default), rag or agent.
.PARAMETER Foreground
    Run a single service in this window instead (used by the scheduled tasks). Needs -Service rag or agent.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File ps\Start-NpcRag.ps1
#>
param(
    [ValidateSet("all", "rag", "agent")][string]$Service = "all",
    [switch]$Foreground
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv\Scripts\python.exe"
$Logs = Join-Path $Root "logs"
if (-not (Test-Path $Venv)) { throw "Run ps\Install-NpcRag.ps1 first." }
New-Item -ItemType Directory -Force $Logs | Out-Null

function Get-Port([string]$Name, [int]$Default) {
    # Read AGENT_PORT / RAG_PORT from the environment or .env so health checks hit the right port.
    $value = [Environment]::GetEnvironmentVariable($Name)
    $envFile = Join-Path $Root ".env"
    if (-not $value -and (Test-Path $envFile)) {
        $line = Select-String -Path $envFile -Pattern "^\s*$Name\s*=\s*(\d+)" | Select-Object -First 1
        if ($line) { $value = $line.Matches[0].Groups[1].Value }
    }
    if ($value) { return [int]$value } else { return $Default }
}

function Wait-Healthy([string]$Name, [int]$Port, [int]$Seconds = 120) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3
            Write-Host "$Name is up on port $Port" -ForegroundColor Green
            return $r
        } catch { Start-Sleep -Seconds 1 }  # Model loading takes a few seconds on first start.
    }
    throw "$Name did not become healthy on port $Port; see $Logs\$Name.log"
}

if ($Foreground) {
    if ($Service -eq "all") { throw "-Foreground needs -Service rag or -Service agent" }
    Set-Location $Root
    & $Venv -m npc_rag serve $Service    # Blocks until the service stops.
    exit $LASTEXITCODE
}

$targets = if ($Service -eq "all") { @("rag", "agent") } else { @($Service) }  # RAG first: the agent depends on it.
foreach ($name in $targets) {
    $port = if ($name -eq "rag") { Get-Port "RAG_PORT" 9092 } else { Get-Port "AGENT_PORT" 9091 }
    $busy = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($busy) { Write-Host "Port $port already in use (PID $($busy[0].OwningProcess)); skipping $name." -ForegroundColor Yellow; continue }
    Start-Process -FilePath $Venv -ArgumentList "-m", "npc_rag", "serve", $name -WorkingDirectory $Root `
        -WindowStyle Hidden -RedirectStandardOutput "$Logs\$name.out.log" -RedirectStandardError "$Logs\$name.log"
    $health = Wait-Healthy $name $port
    if ($name -eq "agent" -and -not $health.llm) {
        Write-Host "Warning: the agent cannot reach llama.cpp at $($health.llm_url). Replies will use fallback lines." -ForegroundColor Yellow
    }
}
