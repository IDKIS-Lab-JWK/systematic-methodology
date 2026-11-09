[CmdletBinding()]
param(
    [ValidateSet("grid", "cv", "test", "all")]
    [string]$Stage = "all",
    [ValidateSet(20, 40, 60, 80, 100)]
    [int]$Percentage = 20,
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
    [string]$CacheDir = "",
    [string]$CacheNamespace = "flexict_vlm_lps_2mm_160cube_depth20_meanhw_864_fp16_v1",
    [string]$Device = "cuda",
    [ValidateSet("auto", "bf16", "fp32")]
    [string]$TrainingPrecision = "auto",
    [ValidateRange(1, 64)]
    [int]$MicroBatchSize = 16,
    [ValidateRange(1, 1000000)]
    [int]$CheckpointInterval = 10,
    [int]$Seed = 42,
    [switch]$ValidateCacheShards,
    [switch]$NoResume,
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

$python = Join-Path $PSScriptRoot ".venv_flexict\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "FlexiCT Python environment not found: $python"
}
$arguments = @(
    "Code\Models\FlexiCT\train_attention_readout.py",
    "--stage", $Stage,
    "--percentage", $Percentage,
    "--parameter_source", $ParameterSource,
    "--model_source", $ModelSource,
    "--cache_namespace", $CacheNamespace,
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
if ($CacheDir) { $arguments += @("--cache_dir", $CacheDir) }
if ($ValidateCacheShards) { $arguments += "--validate_cache_shards" }
if ($NoResume) { $arguments += "--no-resume" }
if ($DryRun) { $arguments += "--dry_run" }

Push-Location $PSScriptRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "FlexiCT frozen attention readout failed with exit code $LASTEXITCODE." }
}
finally { Pop-Location }
