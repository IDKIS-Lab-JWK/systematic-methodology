[CmdletBinding()]
param(
    [string]$DatasetRoot = "",
    [string]$ResultsRoot = "",
    [string]$PromptConfig = "",
    [string]$UpstreamRoot = "",
    [string]$Checkpoint = "",
    [string]$CheckpointSha256 = "bb4e0a697380f3636723b76b175cd1d6ca2e72659a4c269be3b3e5d41c4b7403",
    [string]$HFCacheRoot = "",
    [string]$ProvisionLog = "",
    [string]$Device = "cuda",
    [ValidateSet("auto", "bf16", "fp16", "fp32")]
    [string]$Precision = "auto",
    [switch]$LocalFilesOnly,
    [switch]$AllowTrustedLegacyCheckpoint,
    [switch]$NoResume,
    [switch]$DryRun
)

$expectedCheckpointSha256 = "bb4e0a697380f3636723b76b175cd1d6ca2e72659a4c269be3b3e5d41c4b7403"
if ($CheckpointSha256 -ne $expectedCheckpointSha256) {
    throw "FlexiCT zero-shot requires the audited official checkpoint SHA-256 $expectedCheckpointSha256."
}

$python = Join-Path $PSScriptRoot ".venv_flexict\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "FlexiCT Python environment not found: $python. Run FlexiCT_Environment_Setup.ps1 first."
}
$arguments = @(
    "Code\Models\FlexiCT\run_zero_shot.py",
    "--device", $Device,
    "--precision", $Precision,
    "--checkpoint_sha256", $CheckpointSha256
)
if ($DatasetRoot) { $arguments += @("--dataset_root", $DatasetRoot) }
if ($ResultsRoot) { $arguments += @("--results_root", $ResultsRoot) }
if ($PromptConfig) { $arguments += @("--prompt_config", $PromptConfig) }
if ($UpstreamRoot) { $arguments += @("--upstream_root", $UpstreamRoot) }
if ($Checkpoint) { $arguments += @("--checkpoint", $Checkpoint) }
if ($HFCacheRoot) { $arguments += @("--hf_cache_root", $HFCacheRoot) }
if ($ProvisionLog) { $arguments += @("--provision_log", $ProvisionLog) }
if ($LocalFilesOnly) { $arguments += "--local_files_only" }
if ($AllowTrustedLegacyCheckpoint) { $arguments += "--allow_trusted_legacy_checkpoint" }
if ($NoResume) { $arguments += "--no-resume" }
if ($DryRun) { $arguments += "--dry_run" }

Push-Location $PSScriptRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "FlexiCT zero-shot evaluation failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
