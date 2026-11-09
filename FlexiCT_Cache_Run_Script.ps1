[CmdletBinding()]
param(
    [ValidateSet("plan", "prepare", "validate")]
    [string]$Action = "plan",

    [ValidateSet(100)]
    [int]$Percentage = 100,

    [string]$DatasetRoot = "",
    [string]$ResultsRoot = "",
    [string]$CacheRoot = "",
    [string]$CacheNamespace = "flexict_vlm_lps_2mm_160cube_clsmean_proj_l2_v1",
    [string]$UpstreamRoot = "",
    [string]$UpstreamRevision = "",
    [string]$Checkpoint = "",
    [string]$CheckpointSha256 = "",
    [string]$ProvisionLog = "",
    [string]$Device = "cuda",

    [ValidateSet("auto", "bf16", "fp16", "fp32")]
    [string]$FeaturePrecision = "auto",

    [ValidateSet("stat", "sha256")]
    [string]$SourceFingerprint = "sha256",

    [ValidateSet(0)]
    [int]$NumWorkers = 0,

    [switch]$LocalFilesOnly,
    [switch]$AllowTrustedLegacyCheckpoint,
    [switch]$DryRun
)

if ($AllowTrustedLegacyCheckpoint -and -not $CheckpointSha256) {
    throw "-AllowTrustedLegacyCheckpoint requires -CheckpointSha256."
}

$python = Join-Path $PSScriptRoot ".venv_flexict\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "FlexiCT Python environment not found: $python. Run FlexiCT_Environment_Setup.ps1 first."
}

$pythonArguments = @(
    "Code\Models\FlexiCT\prepare_cache.py",
    "--action", $Action,
    "--percentage", $Percentage,
    "--cache_namespace", $CacheNamespace,
    "--device", $Device,
    "--feature_precision", $FeaturePrecision,
    "--source_fingerprint", $SourceFingerprint,
    "--num_workers", $NumWorkers
)
if ($DatasetRoot) { $pythonArguments += @("--dataset_root", $DatasetRoot) }
if ($ResultsRoot) { $pythonArguments += @("--results_root", $ResultsRoot) }
if ($CacheRoot) { $pythonArguments += @("--cache_root", $CacheRoot) }
if ($UpstreamRoot) { $pythonArguments += @("--upstream_root", $UpstreamRoot) }
if ($UpstreamRevision) { $pythonArguments += @("--upstream_revision", $UpstreamRevision) }
if ($Checkpoint) { $pythonArguments += @("--checkpoint", $Checkpoint) }
if ($CheckpointSha256) { $pythonArguments += @("--checkpoint_sha256", $CheckpointSha256) }
if ($ProvisionLog) { $pythonArguments += @("--provision_log", $ProvisionLog) }
if ($LocalFilesOnly) { $pythonArguments += "--local_files_only" }
if ($AllowTrustedLegacyCheckpoint) { $pythonArguments += "--allow_trusted_legacy_checkpoint" }
if ($DryRun) { $pythonArguments += "--dry_run" }

Push-Location $PSScriptRoot
try {
    & $python @pythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "FlexiCT cache command failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
