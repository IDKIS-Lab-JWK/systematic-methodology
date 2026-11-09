[CmdletBinding()]
param(
    [ValidateSet("final_block", "final_stage", "both")]
    [string]$AdaptationScope = "final_block",
    [ValidateSet("grid", "cv", "test", "all")]
    [string]$Stage = "test",
    [ValidateSet(20, 40, 60, 80, 100)]
    [int]$Percentage = 100,
    [ValidateSet("publication", "grid", "default")]
    [string]$ParameterSource = "publication",
    [string]$ParameterConfig = "",
    [string]$GridConfig = "",
    [ValidateRange(0, 1000)]
    [int]$MaxGridCandidates = 0,
    [ValidateSet("refit_full_development", "cv_ensemble", "best_cv_fold")]
    [string]$ModelSource = "best_cv_fold",
    [string]$DatasetRoot = "",
    [string]$ResultsRoot = "",
    [string]$CacheRoot = "",
    [ValidateSet("frozen", "cache", "nifti")]
    [string]$InputSource = "frozen",
    [string]$UpstreamRoot = "",
    [string]$Checkpoint = "",
    [string]$ProvisionLog = "",
    [string]$Device = "cuda",
    [ValidateSet("auto", "bf16", "fp32")]
    [string]$TrainingPrecision = "auto",
    # Omitted values are resolved from the selected profile by the Python CLI.
    [ValidateRange(1, 2147483647)]
    [Nullable[int]]$MicroBatchSize = $null,
    [ValidateRange(1, 1000000)]
    [int]$CheckpointInterval = 100,
    [int]$Seed = 42,
    [switch]$LocalFilesOnly,
    [switch]$NoResume,
    [switch]$NoPrefetch,
    [switch]$DryRun
)

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

$python = Join-Path $PSScriptRoot ".venv_colipri\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "COLIPRI Python environment not found: $python"
}

$arguments = @(
    "Code\Models\COLIPRI\train_primus_m_partial_finetune.py",
    "--adaptation_scope", $AdaptationScope,
    "--input_source", $InputSource,
    "--stage", $Stage,
    "--percentage", $Percentage,
    "--parameter_source", $ParameterSource,
    "--model_source", $ModelSource,
    "--device", $Device,
    "--training_precision", $TrainingPrecision,
    "--checkpoint_interval", $CheckpointInterval,
    "--seed", $Seed
)
if ($null -ne $MicroBatchSize) { $arguments += @("--micro_batch_size", $MicroBatchSize) }
if ($ParameterConfig) { $arguments += @("--parameter_config", $ParameterConfig) }
if ($GridConfig) { $arguments += @("--grid_config", $GridConfig) }
if ($MaxGridCandidates -gt 0) { $arguments += @("--max_grid_candidates", $MaxGridCandidates) }
if ($DatasetRoot) { $arguments += @("--dataset_root", $DatasetRoot) }
if ($ResultsRoot) { $arguments += @("--results_root", $ResultsRoot) }
if ($CacheRoot) { $arguments += @("--cache_root", $CacheRoot) }
if ($UpstreamRoot) { $arguments += @("--upstream_root", $UpstreamRoot) }
if ($Checkpoint) { $arguments += @("--checkpoint", $Checkpoint) }
if ($ProvisionLog) { $arguments += @("--provision_log", $ProvisionLog) }
if ($LocalFilesOnly) { $arguments += "--local_files_only" }
if ($NoResume) { $arguments += "--no-resume" }
if ($NoPrefetch) { $arguments += "--no-prefetch" }
if ($DryRun) { $arguments += "--dry_run" }

Push-Location $PSScriptRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "COLIPRI Primus-M $AdaptationScope fine-tuning failed with exit code $LASTEXITCODE." }
}
finally { Pop-Location }
