<#
.SYNOPSIS
    Enables the bare 'helpdesk' command in the current PowerShell session.
.DESCRIPTION
    Dot-source from the repository root:
        Set-ExecutionPolicy -Scope Process Bypass -Force; . .\setup-helpdesk.ps1
    Registers a project-local global PowerShell function named 'helpdesk' without
    modifying your PowerShell profile, global PATH, or system settings.
#>
$script:HelpDeskRepoRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$script:HelpDeskLauncher = Join-Path $script:HelpDeskRepoRoot 'helpdesk.ps1'

if (-not (Test-Path -LiteralPath $script:HelpDeskLauncher)) {
    Write-Error "Missing launcher script: $script:HelpDeskLauncher"
    return
}

$global:RailwayHelpDeskLauncherPath = $script:HelpDeskLauncher

function global:helpdesk {
    if ($MyInvocation.ExpectingInput) {
        $input | & $global:RailwayHelpDeskLauncherPath @args
    } else {
        & $global:RailwayHelpDeskLauncherPath @args
    }
}

Write-Host "Railway Passenger Help Desk command 'helpdesk' enabled for this PowerShell session." -ForegroundColor Cyan
Write-Host "Run 'helpdesk' to open the interactive menu (no PATH or PowerShell profile changes were made)." -ForegroundColor Gray
