<#
.SYNOPSIS
    Project-local launcher for the Railway Passenger Help Desk.
.DESCRIPTION
    Reuses the existing app\.venv virtual environment on subsequent launches,
    or guides first-time setup / dependency recovery with Python 3.12 without
    modifying global Python packages, PATH, or the user's PowerShell profile.
#>
$LauncherArgs = @($args)

$ErrorActionPreference = 'Stop'

try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    [Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {
}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$AppDir = Join-Path $ScriptDir 'app'
$VenvDir = Join-Path $AppDir '.venv'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$RequirementsFile = Join-Path $AppDir 'requirements.txt'

function Find-CompatiblePython {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $probe = "import sys; print(sys.executable) if sys.version_info[:2] == (3, 12) else sys.exit(1)"
        if (Get-Command py -ErrorAction SilentlyContinue) {
            $out = & py -3.12 -c $probe 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) {
                $candidate = ($out | Select-Object -First 1).Trim()
                if (Test-Path -LiteralPath $candidate) { return $candidate }
            }
        }
        foreach ($cmdName in @('python3.12', 'python')) {
            if (Get-Command $cmdName -ErrorAction SilentlyContinue) {
                $out = & $cmdName -c $probe 2>$null
                if ($LASTEXITCODE -eq 0 -and $out) {
                    $candidate = ($out | Select-Object -First 1).Trim()
                    if (Test-Path -LiteralPath $candidate) { return $candidate }
                }
            }
        }
        return $null
    } finally {
        $ErrorActionPreference = $prev
    }
}

function Test-VenvPython {
    param([string]$PyExe)
    if (-not (Test-Path -LiteralPath $PyExe)) { return $false }
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $PyExe -c "import importlib.util, sys; mods=('chromadb','sentence_transformers','transformers','torch','rank_bm25','numpy','openai','pydantic','dotenv','pymupdf','bs4','rapidocr_onnxruntime','rich'); sys.exit(0 if (sys.version_info[:2] == (3, 12) and all(importlib.util.find_spec(m) is not None for m in mods)) else 1)" 2>$null
        return ($LASTEXITCODE -eq 0)
    } finally {
        $ErrorActionPreference = $prev
    }
}

$script:LastAppExit = 0

$HasStdinPipe = $MyInvocation.ExpectingInput
$StdinItems = if ($HasStdinPipe) { @($input) } else { @() }

function Invoke-HelpDeskModule {
    param(
        [string]$PyExe,
        [string[]]$ModuleArgs
    )
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    Push-Location -LiteralPath $AppDir
    try {
        if ($ModuleArgs -and $ModuleArgs.Count -gt 0) {
            if ($HasStdinPipe) {
                $StdinItems | & $PyExe -m helpdesk.cli @ModuleArgs
            } else {
                & $PyExe -m helpdesk.cli @ModuleArgs
            }
        } else {
            if ($HasStdinPipe) {
                $StdinItems | & $PyExe -m helpdesk.cli
            } else {
                & $PyExe -m helpdesk.cli
            }
        }
        $script:LastAppExit = $LASTEXITCODE
    } finally {
        Pop-Location
        $ErrorActionPreference = $prev
    }
}

if (-not (Test-Path -LiteralPath $RequirementsFile)) {
    Write-Host "[ERROR] Could not find app\requirements.txt under $ScriptDir" -ForegroundColor Red
    exit 1
}

$ForceSetup = $false
$CliPassArgs = @()
foreach ($arg in ($LauncherArgs | Where-Object { $_ -ne $null -and $_ -ne '' })) {
    if ($arg -in @('--setup', '-Setup', '-AutoSetup')) {
        $ForceSetup = $true
    } else {
        $CliPassArgs += $arg
    }
}

# Fast path: reuse existing configured virtual environment cleanly without reinstalling anything.
if ((Test-Path -LiteralPath $VenvPython) -and (-not $ForceSetup)) {
    if (Test-VenvPython -PyExe $VenvPython) {
        Invoke-HelpDeskModule -PyExe $VenvPython -ModuleArgs $CliPassArgs
        exit $script:LastAppExit
    }
    Write-Host "Notice: app\.venv exists, but required Python 3.12 packages are missing or incomplete." -ForegroundColor Yellow
    Write-Host "Opening setup/repair menu without deleting your existing environment..." -ForegroundColor Cyan
    $RepairPy = Find-CompatiblePython
    if (-not $RepairPy) { $RepairPy = $VenvPython }
    Invoke-HelpDeskModule -PyExe $RepairPy -ModuleArgs @('--setup-menu')
    exit $script:LastAppExit
}

# First-time setup or explicit --setup repair path.
$BasePython = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { Find-CompatiblePython }
if (-not $BasePython) {
    Write-Host "[FAIL] Python 3.12 is required but was not detected on PATH or via the Python Launcher (py -3.12)." -ForegroundColor Red
    Write-Host "Please install Python 3.12 (x64) from python.org and re-run 'helpdesk'." -ForegroundColor Yellow
    exit 1
}

if ($ForceSetup) {
    Invoke-HelpDeskModule -PyExe $BasePython -ModuleArgs @('--setup')
    exit $script:LastAppExit
}

Invoke-HelpDeskModule -PyExe $BasePython -ModuleArgs @('--setup-menu')
exit $script:LastAppExit
