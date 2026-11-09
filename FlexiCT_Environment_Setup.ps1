[CmdletBinding()]
param(
    [ValidateSet("create", "install", "all")]
    [string]$Action = "all",

    [string]$BootstrapPython = "",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",

    [switch]$SkipPipUpgrade,
    [switch]$DryRun
)

$environmentRoot = Join-Path $PSScriptRoot ".venv_flexict"
$environmentPython = Join-Path $environmentRoot "Scripts\python.exe"
$requirements = Join-Path $PSScriptRoot "Code\Models\FlexiCT\requirements.txt"

if (-not (Test-Path -LiteralPath $requirements -PathType Leaf)) {
    throw "FlexiCT requirements file not found: $requirements"
}

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    if ($DryRun) {
        Write-Host ("DRY RUN: {0} {1}" -f $Executable, ($Arguments -join " "))
        return
    }
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable"
    }
}

function Confirm-CPython311 {
    Invoke-CheckedCommand -Executable $environmentPython -Arguments @(
        "-c",
        "import sys; assert sys.implementation.name == 'cpython' and sys.version_info[:2] == (3, 11), f'Requires CPython 3.11; found {sys.implementation.name} {sys.version}'"
    )
}

Push-Location $PSScriptRoot
try {
    if ($Action -in @("create", "all")) {
        if (Test-Path -LiteralPath $environmentPython -PathType Leaf) {
            Write-Host "Reusing existing FlexiCT environment: $environmentRoot"
        }
        else {
            if ($BootstrapPython) {
                Invoke-CheckedCommand -Executable $BootstrapPython -Arguments @(
                    "-m", "venv", $environmentRoot
                )
            }
            elseif ($DryRun) {
                if (-not (Get-Command "py" -ErrorAction SilentlyContinue)) {
                    Write-Warning "The py launcher is not currently available. Supply -BootstrapPython C:\Path\To\python.exe for the real run."
                }
                Invoke-CheckedCommand -Executable "py" -Arguments @(
                    "-3.11", "-m", "venv", $environmentRoot
                )
            }
            else {
                $pyLauncher = Get-Command "py" -ErrorAction SilentlyContinue
                if (-not $pyLauncher) {
                    throw "Python launcher 'py' was not found. Install CPython 3.11 with the Windows launcher or pass -BootstrapPython C:\Path\To\python.exe."
                }
                Invoke-CheckedCommand -Executable $pyLauncher.Source -Arguments @(
                    "-3.11", "-m", "venv", $environmentRoot
                )
            }
        }
        if ($Action -eq "create") {
            Confirm-CPython311
        }
    }

    if ($Action -in @("install", "all")) {
        if (-not $DryRun -and -not (Test-Path -LiteralPath $environmentPython -PathType Leaf)) {
            throw "FlexiCT environment not found: $environmentPython. Run with -Action create or all first."
        }
        Confirm-CPython311
        if (-not $SkipPipUpgrade) {
            Invoke-CheckedCommand -Executable $environmentPython -Arguments @(
                "-m", "pip", "install", "--upgrade", "pip"
            )
        }
        Invoke-CheckedCommand -Executable $environmentPython -Arguments @(
            "-m", "pip", "install", "--index-url", $TorchIndexUrl,
            "torch==2.8.0", "torchvision==0.23.0"
        )
        Invoke-CheckedCommand -Executable $environmentPython -Arguments @(
            "-m", "pip", "install", "-r", $requirements
        )
        Invoke-CheckedCommand -Executable $environmentPython -Arguments @(
            "Code\Models\FlexiCT\check_setup.py", "--dry_run"
        )
    }

    if (-not $DryRun -and (Test-Path -LiteralPath $environmentPython -PathType Leaf)) {
        Invoke-CheckedCommand -Executable $environmentPython -Arguments @("--version")
        Write-Host "FlexiCT environment is available at $environmentRoot"
    }
}
finally {
    Pop-Location
}
