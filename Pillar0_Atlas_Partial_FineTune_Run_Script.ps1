[CmdletBinding()]
param(
    [ValidateSet("final_block", "final_stage", "both")]
    [string]$AdaptationScope = "final_block",
    [ValidateSet("grid", "cv", "test", "all")]
    [string]$Stage = "all",
    [ValidateSet(20, 40, 60, 80, 100)]
    [int]$Percentage = 100,
    [ValidateSet("publication", "grid", "default")]
    [string]$ParameterSource = "publication",
    [string]$ParameterConfig = "",
    [string]$GridConfig = "",
    [ValidateRange(0, 1000)]
    [int]$MaxGridCandidates = 0,
    [ValidateSet("refit_full_development", "cv_ensemble", "best_cv_fold")]
    [string]$ModelSource = "cv_ensemble",
    [string]$DatasetRoot = "",
    [string]$ResultsRoot = "",
    [string]$CacheDir = "",
    [string]$CacheNamespace = "pillar0_abdct_rave_1p25mm_384cube_rawhu_lz4_v1",
    [ValidateSet("frozen", "cache")]
    [string]$InputSource = "frozen",
    [string]$FrozenCacheRoot = "",
    [string]$Device = "cuda",
    [ValidateSet("auto", "bf16", "fp32")]
    [string]$TrainingPrecision = "auto",
    [ValidateRange(1, 32)]
    [int]$MicroBatchSize = 1,
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
if ($ParameterSource -eq "grid" -and $ParameterConfig) {
    throw "Grid source does not accept -ParameterConfig."
}
if ($ParameterSource -eq "default" -and ($GridConfig -or $MaxGridCandidates -gt 0)) {
    throw "Default source does not accept grid controls."
}

$python = Join-Path $PSScriptRoot ".venv_pillar0\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Pillar-0 Python environment not found: $python"
}

$arguments = @(
    "Code\Models\Pillar0\train_atlas_partial_finetune.py",
    "--input_source", $InputSource,
    "--adaptation_scope", $AdaptationScope,
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
if ($FrozenCacheRoot) { $arguments += @("--frozen_cache_root", $FrozenCacheRoot) }
if ($LocalFilesOnly) { $arguments += "--local_files_only" }
if ($NoResume) { $arguments += "--no-resume" }
if ($NoPrefetch) { $arguments += "--no-prefetch" }
if ($DryRun) { $arguments += "--dry_run" }

Push-Location $PSScriptRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Pillar-0 Atlas $AdaptationScope fine-tuning failed with exit code $LASTEXITCODE." }
}
finally { Pop-Location }
