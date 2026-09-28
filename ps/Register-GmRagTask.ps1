<# Run as Administrator. Keep the GM listener restricted to the game server's public egress IP. #>
param([string[]]$AllowFrom = @(), [int]$Port = 9093, [switch]$Remove)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$StartScript = Join-Path $PSScriptRoot 'Start-GmRag.ps1'
$TaskName = 'npc-rag GM help'
if ($Remove) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Remove-NetFirewallRule -DisplayName $TaskName -ErrorAction SilentlyContinue
    return # Never touches the existing player RAG service or its firewall rule.
}
$Action = New-ScheduledTaskAction -Execute 'powershell.exe' -WorkingDirectory $ProjectRoot -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$StartScript`""
$Trigger = New-ScheduledTaskTrigger -AtStartup
$Trigger.Delay = 'PT30S'
$Settings = New-ScheduledTaskSettingsSet -RestartCount 99 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable
$Principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Force | Out-Null
if ($AllowFrom.Count -gt 0) {
    Remove-NetFirewallRule -DisplayName $TaskName -ErrorAction SilentlyContinue
    New-NetFirewallRule -DisplayName $TaskName -Direction Inbound -Protocol TCP -LocalPort $Port -RemoteAddress $AllowFrom -Action Allow | Out-Null
} # Omit AllowFrom to retain existing firewall policy; never adds an unrestricted inbound rule.
Start-ScheduledTask -TaskName $TaskName
