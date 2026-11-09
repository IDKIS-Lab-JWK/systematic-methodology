[CmdletBinding()]
param(
    [ValidateSet("grid", "cv", "test", "all")]
    [string]$Stage = "all",
    [ValidateSet(20, 40, 60, 80, 100)]
    [int]$Percentage = 40,
    [ValidateSet("publication", "grid", "default")]
    [string]$ParameterSource = "publication",
    [string]$ParameterConfig = "",
    [string]$GridConfig = "",
    [ValidateRange(0, 1000)]
    [int]$MaxGridCandidates = 0,
    [ValidateSet("refit_full_development", "cv_ensemble", "best_cv_fold")]
    [string]$ModelSource = "refit_full_development",
    [string]$DatasetRoot = "",
    [string]$ResultsRoot = "",
    [string]$UpstreamRoot = "",
    [string]$Checkpoint = "",
    [string]$CheckpointSha256 = "",
    [string]$ProvisionLog = "",
    [string]$Device = "cuda",
    [ValidateSet("auto", "bf16", "fp32")]
    [string]$TrainingPrecision = "auto",
    [ValidateRange(1, 8)]
    [int]$MicroBatchSize = 1,
    [ValidateRange(1, 1000000)]
    [int]$CheckpointInterval = 100,
    [int]$Seed = 42,
    [switch]$LocalFilesOnly,
    [switch]$AllowTrustedLegacyCheckpoint,
    [switch]$NoResume,
    [switch]$DryRun
)

if ($AllowTrustedLegacyCheckpoint -and -not $CheckpointSha256) {
    throw "-AllowTrustedLegacyCheckpoint requires -CheckpointSha256."
}
if ($Stage -eq "grid" -and $ParameterSource -notin @("publication", "grid")) {
    throw "-Stage $Stage requires -ParameterSource publication or grid."
}
if ($ParameterSource -eq "publication" -and ($ParameterConfig -or $GridConfig -or $MaxGridCandidates -gt 0)) {
    throw "Publication source uses the packaged complete profile without overrides."
}
if ($ParameterSource -eq "grid" -and $ParameterConfig) { throw "Grid source does not accept -ParameterConfig." }
if ($ParameterSource -eq "default" -and ($GridConfig -or $MaxGridCandidates -gt 0)) {
    throw "Default source does not accept grid controls."
}

$python = Join-Path $PSScriptRoot ".venv_flexict\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "FlexiCT Python environment not found: $python"
}
$arguments = @(
    "Code\Models\FlexiCT\train_final_quarter_finetune.py",
    "--stage", $Stage,
    "--percentage", $Percentage,
    "--parameter_source", $ParameterSource,
    "--model_source", $ModelSource,
    "--device", $Device,
    "--training_precision", $TrainingPrecision,
    "--micro_batch_size", $MicroBatchSize,
    "--checkpoint_interval", $CheckpointInterval,
    "--seed", $Seed
)
if ($ParameterConfig) { $arguments += @("--parameter_config", $ParameterConfig) }
if ($GridConfig) { $arguments += @("--grid_config", $GridConfig) }
if ($MaxGridCandidates -gt 0) { $arguments += @("--max_grid_candidates", $MaxGridCandidates) }
if ($DatasetRoot) { $arguments += @("--dataset_root", $DatasetRoot) }
if ($ResultsRoot) { $arguments += @("--results_root", $ResultsRoot) }
if ($UpstreamRoot) { $arguments += @("--upstream_root", $UpstreamRoot) }
if ($Checkpoint) { $arguments += @("--checkpoint", $Checkpoint) }
if ($CheckpointSha256) { $arguments += @("--checkpoint_sha256", $CheckpointSha256) }
if ($ProvisionLog) { $arguments += @("--provision_log", $ProvisionLog) }
if ($LocalFilesOnly) { $arguments += "--local_files_only" }
if ($AllowTrustedLegacyCheckpoint) { $arguments += "--allow_trusted_legacy_checkpoint" }
if ($NoResume) { $arguments += "--no-resume" }
if ($DryRun) { $arguments += "--dry_run" }

Push-Location $PSScriptRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "FlexiCT final-quarter fine-tuning failed with exit code $LASTEXITCODE." }
}
finally { Pop-Location }
