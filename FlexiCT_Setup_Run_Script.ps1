[CmdletBinding()]
param(
    [string]$UpstreamRoot = "",
    [string]$UpstreamRevision = "",
    [string]$Checkpoint = "",
    [string]$CheckpointSha256 = "",
    [string]$ProvisionLog = "",
    [string]$Device = "cuda",

    [switch]$LocalFilesOnly,
    [switch]$AllowTrustedLegacyCheckpoint,
    [switch]$LoadModel,
    [switch]$DryRun
)

if ($AllowTrustedLegacyCheckpoint -and -not $CheckpointSha256) {
    throw "-AllowTrustedLegacyCheckpoint requires -CheckpointSha256."
}

$python = Join-Path $PSScriptRoot ".venv_flexict\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "FlexiCT Python environment not found: $python. Run FlexiCT_Environment_Setup.ps1 first."
}

$pythonArguments = @("Code\Models\FlexiCT\check_setup.py", "--device", $Device)
if ($UpstreamRoot) { $pythonArguments += @("--upstream_root", $UpstreamRoot) }
if ($UpstreamRevision) { $pythonArguments += @("--upstream_revision", $UpstreamRevision) }
if ($Checkpoint) { $pythonArguments += @("--checkpoint", $Checkpoint) }
if ($CheckpointSha256) { $pythonArguments += @("--checkpoint_sha256", $CheckpointSha256) }
if ($ProvisionLog) { $pythonArguments += @("--provision_log", $ProvisionLog) }
if ($LocalFilesOnly) { $pythonArguments += "--local_files_only" }
if ($AllowTrustedLegacyCheckpoint) { $pythonArguments += "--allow_trusted_legacy_checkpoint" }
if ($LoadModel) { $pythonArguments += "--load_model" }
if ($DryRun) { $pythonArguments += "--dry_run" }

Push-Location $PSScriptRoot
try {
    & $python @pythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "FlexiCT setup check failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
