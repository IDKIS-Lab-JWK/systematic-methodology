[CmdletBinding()]
param(
    [ValidateSet("grid", "cv", "test", "all")]
    [string]$Stage = "all",

    [ValidateSet(20, 40, 60, 80, 100)]
    [int]$Percentage = 40,

    [ValidateSet("publication", "grid", "default", "manual")]
    [string]$ParameterSource = "publication",

    [string]$ParameterConfig = "",
    [string]$GridConfig = "",

    [ValidateRange(0, 1000)]
    [int]$MaxGridCandidates = 0,

    [ValidateSet("refit_full_development", "cv_ensemble", "best_cv_fold")]
    [string]$ModelSource = "refit_full_development",

    [string]$DatasetRoot = "",
    [string]$ResultsRoot = "",
    [string]$CacheRoot = "",
    [string]$CacheNamespace = "colipri_ras_2mm_192cube_raw_backbone_864x24cube_fp32_v1",
    [string]$UpstreamRoot = "",
    [string]$UpstreamRevision = "",
    [string]$Checkpoint = "",
    [string]$ProvisionLog = "",
    [string]$Device = "cuda",

    [ValidateSet("auto", "bf16", "fp16", "fp32")]
    [string]$FeaturePrecision = "auto",

    [ValidateSet("auto", "bf16", "fp32")]
    [string]$TrainingPrecision = "auto",

    [ValidateSet("stat", "sha256")]
    [string]$SourceFingerprint = "sha256",

    [ValidateRange(1, 2147483647)]
    [int]$MicroBatchSize = 16,

    [ValidateRange(1, 1000000)]
    [int]$CheckpointInterval = 250,

    [int]$Seed = 42,

    [ValidateSet(0)]
    [int]$NumWorkers = 0,

    [switch]$LocalFilesOnly,
    [switch]$NoResume,
    [switch]$NoPrefetch,
    [switch]$DryRun,
    [switch]$ForceTest
)

if ($MicroBatchSize -gt 16) {
    throw "Selected microbatch size $MicroBatchSize exceeds the effective batch size (16). Select a microbatch size equal to or lower than 16."
}
if (16 % $MicroBatchSize -ne 0) {
    throw "Selected microbatch size $MicroBatchSize must divide the effective batch size (16) evenly. Select one of: 1, 2, 4, 8, 16."
}

if (-not $DryRun -and $SourceFingerprint -ne "sha256") {
    throw "COLIPRI pooling-optimised probe execution requires -SourceFingerprint sha256; stat is cache diagnostics/dry-run only."
}

if ($ParameterSource -eq "manual" -and -not $ParameterConfig) {
    throw "-ParameterSource manual requires -ParameterConfig."
}
if ($Stage -in @("grid", "all") -and $ParameterSource -notin @("publication", "grid")) {
    throw "-Stage $Stage requires -ParameterSource publication or grid."
}
if ($ParameterSource -eq "publication" -and $ParameterConfig) {
    throw "-ParameterSource publication does not accept -ParameterConfig; the packaged 20-candidate grid is fixed."
}
if ($ParameterSource -eq "publication" -and $GridConfig) {
    throw "-ParameterSource publication does not accept -GridConfig; the audited 20-candidate grid is fixed."
}
if ($ParameterSource -eq "publication" -and $MaxGridCandidates -gt 0) {
    throw "-ParameterSource publication does not accept -MaxGridCandidates; all 20 candidates are required."
}
if ($ParameterSource -eq "grid" -and $ParameterConfig) {
    throw "-ParameterSource grid does not accept -ParameterConfig."
}
if ($ParameterSource -in @("default", "manual") -and ($GridConfig -or $MaxGridCandidates -gt 0)) {
    throw "Fixed default/manual parameter sources do not accept -GridConfig or -MaxGridCandidates."
}

$python = Join-Path $PSScriptRoot ".venv_colipri\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "COLIPRI Python environment not found: $python. Run COLIPRI_Environment_Setup.ps1 first."
}

$pythonArguments = @(
    "Code\Models\COLIPRI\train_pooling_optimised_linear_probe.py",
    "--stage", $Stage,
    "--percentage", $Percentage,
    "--parameter_source", $ParameterSource,
    "--model_source", $ModelSource,
    "--cache_namespace", $CacheNamespace,
    "--device", $Device,
    "--feature_precision", $FeaturePrecision,
    "--training_precision", $TrainingPrecision,
    "--source_fingerprint", $SourceFingerprint,
    "--micro_batch_size", $MicroBatchSize,
    "--checkpoint_interval", $CheckpointInterval,
    "--seed", $Seed,
    "--num_workers", $NumWorkers
)
if ($ParameterConfig) { $pythonArguments += @("--parameter_config", $ParameterConfig) }
if ($GridConfig) { $pythonArguments += @("--grid_config", $GridConfig) }
if ($MaxGridCandidates -gt 0) { $pythonArguments += @("--max_grid_candidates", $MaxGridCandidates) }
if ($DatasetRoot) { $pythonArguments += @("--dataset_root", $DatasetRoot) }
if ($ResultsRoot) { $pythonArguments += @("--results_root", $ResultsRoot) }
if ($CacheRoot) { $pythonArguments += @("--cache_root", $CacheRoot) }
if ($UpstreamRoot) { $pythonArguments += @("--upstream_root", $UpstreamRoot) }
if ($UpstreamRevision) { $pythonArguments += @("--upstream_revision", $UpstreamRevision) }
if ($Checkpoint) { $pythonArguments += @("--checkpoint", $Checkpoint) }
if ($ProvisionLog) { $pythonArguments += @("--provision_log", $ProvisionLog) }
if ($LocalFilesOnly) { $pythonArguments += "--local_files_only" }
if ($NoResume) { $pythonArguments += "--no-resume" }
if ($NoPrefetch) { $pythonArguments += "--no-prefetch" }
if ($DryRun) { $pythonArguments += "--dry_run" }
if ($ForceTest) { $pythonArguments += "--force_test" }

Push-Location $PSScriptRoot
try {
    & $python @pythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "COLIPRI pooling-optimised probe command failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
