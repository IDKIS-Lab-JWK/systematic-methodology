[CmdletBinding()]
param(
    [string]$UpstreamRoot = "",
    [string]$UpstreamRevision = "",
    [string]$Checkpoint = "",
    [string]$ProvisionLog = "",
    [string]$Device = "cuda",

    [switch]$LocalFilesOnly,
    [switch]$LoadModel,
    [switch]$DryRun
)

$python = Join-Path $PSScriptRoot ".venv_colipri\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "COLIPRI Python environment not found: $python. Run COLIPRI_Environment_Setup.ps1 first."
}

$pythonArguments = @("Code\Models\COLIPRI\check_setup.py", "--device", $Device)
if ($UpstreamRoot) { $pythonArguments += @("--upstream_root", $UpstreamRoot) }
if ($UpstreamRevision) { $pythonArguments += @("--upstream_revision", $UpstreamRevision) }
if ($Checkpoint) { $pythonArguments += @("--checkpoint", $Checkpoint) }
if ($ProvisionLog) { $pythonArguments += @("--provision_log", $ProvisionLog) }
if ($LocalFilesOnly) { $pythonArguments += "--local_files_only" }
if ($LoadModel) { $pythonArguments += "--load_model" }
if ($DryRun) { $pythonArguments += "--dry_run" }

Push-Location $PSScriptRoot
try {
    & $python @pythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "COLIPRI setup check failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
