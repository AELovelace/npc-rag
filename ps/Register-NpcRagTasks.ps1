<#
.SYNOPSIS
    Run npc-rag at boot as a scheduled task (restarting on failure), and optionally open its port to the game server.
    Run from an elevated (Administrator) PowerShell.
.PARAMETER AllowFrom
    IP address(es) allowed to reach the port, e.g. the LiDollQuest game server. Omit to leave the firewall unchanged.
.PARAMETER Port
    npc-rag's port (NPC_PORT). Default 9092.
.PARAMETER Remove
    Delete the task and the firewall rule instead.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File ps\Register-NpcRagTasks.ps1 -AllowFrom 10.1.1.23
#>
param(
    [string[]]$AllowFrom = @(),
    [int]$Port = 9092,
    [switch]$Remove
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Start = Join-Path $PSScriptRoot "Start-NpcRag.ps1"
$TaskName = "npc-rag"
$RuleName = "npc-rag"
# Task and rule names used before the classifier and RAG merged into one service.
$OldTasks = @("npc-rag rag", "npc-rag agent")
$OldRule = "npc-rag agent 9091"

foreach ($name in $OldTasks) { Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue }
Remove-NetFirewallRule -DisplayName $OldRule -ErrorAction SilentlyContinue

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Remove-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue
    Write-Host "Removed the npc-rag task and firewall rule."
    return
}

$action = New-ScheduledTaskAction -Execute "powershell.exe" -WorkingDirectory $Root `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Start`" -Foreground"
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = "PT30S"  # Give Sakura time to start its model servers.
$settings = New-ScheduledTaskSettingsSet -RestartCount 99 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable   # Restart on crash; never time out.
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Host "Registered and started task '$TaskName'."

if ($AllowFrom.Count -gt 0) {
    Remove-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue
    New-NetFirewallRule -DisplayName $RuleName -Direction Inbound -Protocol TCP -LocalPort $Port `
        -RemoteAddress $AllowFrom -Action Allow | Out-Null   # Only the listed addresses.
    Write-Host "Firewall: port $Port open to $($AllowFrom -join ', ')."
}
