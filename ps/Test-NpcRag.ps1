<#
.SYNOPSIS
    Smoke test against the running service: health, one game question, one piece of small talk.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File ps\Test-NpcRag.ps1 -ApiKey "your NPC_API_KEY"
#>
param(
    [string]$Url = "http://127.0.0.1:9092",
    [string]$ApiKey = ""
)
$ErrorActionPreference = "Stop"
$headers = @{}
if ($ApiKey) { $headers["X-Api-Key"] = $ApiKey }

$health = Invoke-RestMethod -Uri "$Url/health" -TimeoutSec 10
Write-Host "npc-rag ok=$($health.ok)  llm=$($health.llm) ($($health.llm_url))  classifier=$($health.classifier) ($($health.classifier_url))"
if (-not $health.ok) { throw "npc-rag has no wiki index." }

foreach ($message in @("how do I change my diaper?", "hi! how are you today?")) {
    $body = @{ message = $message; player_name = "Tester" } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "$Url/v1/npc/chat" -Method Post -Body $body -ContentType "application/json" `
        -Headers $headers -TimeoutSec 120
    Write-Host ""
    Write-Host "PLAYER: $message"
    Write-Host "[$($r.category) via $($r.method), guard=$($r.guard), $($r.timings_ms.total) ms, fallback=$($r.fallback)]"
    Write-Host "$($r.npc_name): $($r.reply)"
    foreach ($s in $r.sources) { Write-Host "  source: $($s.title) > $($s.section)  $($s.url)" }
}
