[CmdletBinding()]
param(
    [ValidateSet(100)]
    [int]$Percentage = 100,
    [string]$DatasetRoot = "",
    [string]$CacheDir = "",
    [string]$CacheNamespace = "flexict_vlm_lps_2mm_160cube_depth20_meanhw_864_fp16_v1",
    [string]$UpstreamRoot = "",
    [string]$Checkpoint = "",
    [string]$CheckpointSha256 = "",
    [string]$ProvisionLog = "",
    [string]$Device = "cuda",
    [switch]$LocalFilesOnly,
    [switch]$AllowTrustedLegacyCheckpoint,
    [switch]$ValidateShards,
    [switch]$NoResume,
    [switch]$DryRun
)

if ($AllowTrustedLegacyCheckpoint -and -not $CheckpointSha256) {
    throw "-AllowTrustedLegacyCheckpoint requires -CheckpointSha256."
}
$python = Join-Path $PSScriptRoot ".venv_flexict\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "FlexiCT Python environment not found: $python"
}
$arguments = @(
    "Code\Models\FlexiCT\prepare_attention_cache.py",
    "--percentage", $Percentage,
    "--cache_namespace", $CacheNamespace,
    "--device", $Device
)
if ($DatasetRoot) { $arguments += @("--dataset_root", $DatasetRoot) }
if ($CacheDir) { $arguments += @("--cache_dir", $CacheDir) }
if ($UpstreamRoot) { $arguments += @("--upstream_root", $UpstreamRoot) }
if ($Checkpoint) { $arguments += @("--checkpoint", $Checkpoint) }
if ($CheckpointSha256) { $arguments += @("--checkpoint_sha256", $CheckpointSha256) }
if ($ProvisionLog) { $arguments += @("--provision_log", $ProvisionLog) }
if ($LocalFilesOnly) { $arguments += "--local_files_only" }
if ($AllowTrustedLegacyCheckpoint) { $arguments += "--allow_trusted_legacy_checkpoint" }
if ($ValidateShards) { $arguments += "--validate_shards" }
if ($NoResume) { $arguments += "--no-resume" }
if ($DryRun) { $arguments += "--dry_run" }

Push-Location $PSScriptRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "FlexiCT attention-token cache command failed with exit code $LASTEXITCODE." }
}
finally { Pop-Location }
