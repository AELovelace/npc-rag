<#
.SYNOPSIS
    Run both services at boot as scheduled tasks (restarting on failure), and optionally open port 9091 to the game server.
    Run from an elevated (Administrator) PowerShell.
.PARAMETER AllowFrom
    IP address(es) allowed to reach port 9091, e.g. the LiDollQuest game server. Omit to leave the firewall unchanged.
.PARAMETER Remove
    Delete the tasks and the firewall rule instead.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File ps\Register-NpcRagTasks.ps1 -AllowFrom 10.1.1.23
#>
param(
    [string[]]$AllowFrom = @(),
    [switch]$Remove
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Start = Join-Path $PSScriptRoot "Start-NpcRag.ps1"
$RuleName = "npc-rag agent 9091"

if ($Remove) {
    foreach ($name in @("npc-rag rag", "npc-rag agent")) {
        Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
    }
    Remove-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue
    Write-Host "Removed npc-rag tasks and firewall rule."
    return
}

foreach ($service in @("rag", "agent")) {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -WorkingDirectory $Root `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Start`" -Service $service -Foreground"
    $trigger = New-ScheduledTaskTrigger -AtStartup
    if ($service -eq "agent") { $trigger.Delay = "PT20S" }  # Give the RAG service a head start at boot.
    $settings = New-ScheduledTaskSettingsSet -RestartCount 99 -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable   # Restart on crash; never time out.
    $principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
    Register-ScheduledTask -TaskName "npc-rag $service" -Action $action -Trigger $trigger -Settings $settings `
        -Principal $principal -Force | Out-Null
    Start-ScheduledTask -TaskName "npc-rag $service"
    Write-Host "Registered and started task 'npc-rag $service'."
    if ($service -eq "rag") { Start-Sleep -Seconds 15 }  # The agent checks RAG health on its first requests.
}

if ($AllowFrom.Count -gt 0) {
    Remove-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue
    New-NetFirewallRule -DisplayName $RuleName -Direction Inbound -Protocol TCP -LocalPort 9091 `
        -RemoteAddress $AllowFrom -Action Allow | Out-Null   # Only the listed addresses; 9092 stays local.
    Write-Host "Firewall: port 9091 open to $($AllowFrom -join ', ')."
}
