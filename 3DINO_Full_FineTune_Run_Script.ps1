[CmdletBinding()]
param(
    # Canonical stages are grid/cv/test/all; "search" is input compatibility only.
    [Parameter(HelpMessage = "Workflow stage: grid, cv, test, or all.")]
    [ValidateScript({
        if ($_ -in @("grid", "search", "cv", "test", "all")) { return $true }
        throw "Stage must be grid, cv, test, or all."
    })]
    [string]$Stage = "all",

    [ValidateSet(20, 40, 60, 80, 100)]
    [int]$Percentage = 40,

    # There is no executable publication full-fine-tuning profile for 3DINO.
    # Canonical sources are default/grid/manual; "search" is legacy.
    [Parameter(HelpMessage = "Parameter source: default, grid, or manual.")]
    [ValidateScript({
        if ($_ -in @("default", "grid", "search", "manual")) { return $true }
        throw "ParameterSource must be default, grid, or manual."
    })]
    [string]$ParameterSource = "default",

    [string]$ParameterConfig = "",
    [Alias("SearchConfig")]
    [string]$GridConfig = "",

    [Alias("MaxSearchCandidates")]
    [ValidateRange(0, 1000)]
    [int]$MaxGridCandidates = 0,

    [ValidateSet("refit_full_development", "cv_ensemble", "best_cv_fold", "provided_checkpoint")]
    [string]$ModelSource = "refit_full_development",

    [string]$Checkpoint = "",

    # Optional output override. Empty preserves the Python CLI default:
    # <methodology root>\Results\3DINO.
    [Parameter(HelpMessage = "Optional output root; empty preserves the methodology Results/3DINO default.")]
    [string]$ResultsRoot = "",

    [string]$CacheRoot = "",
    [string]$CacheNamespace = "threedino_ct_ras_p0005_p9995_fg_resize_144x144x112_v2",
    [string]$ModelRepoId = "",
    [string]$ModelRevision = "",
    [string]$UpstreamRoot = "",
    [string]$UpstreamRevision = "",
    [string]$PretrainedCheckpoint = "",
    [string]$PretrainedCheckpointSha256 = "",
    [string]$ProvisionLog = "",
    [string]$Device = "cuda",

    [int]$Seed = 42,

    # CPU workers for the training loader only. The default remains 0. Values
    # above 0 use persistent workers; validation/test loaders remain fixed at 0.
    # A different value may be selected when restarting from a committed checkpoint.
    [Parameter(HelpMessage = "Training DataLoader workers (default 0); values above 0 are persistent, validation/test stay at 0, and the count may change only at checkpoint restart.")]
    [ValidateRange(0, 64)]
    [int]$NumWorkers = 0,

    [switch]$AllowExperimentalFullFineTune,
    [switch]$LocalFilesOnly,
    [switch]$NoResume,
    [switch]$DryRun,
    [switch]$ForceTest
)

if ($Stage -eq "search") {
    $Stage = "grid"
}
if ($ParameterSource -eq "search") {
    $ParameterSource = "grid"
}
if ($Stage -eq "grid" -and $ParameterSource -eq "default") {
    $ParameterSource = "grid"
}
if ($Stage -eq "grid" -and $ParameterSource -eq "manual") {
    throw "-Stage grid requires -ParameterSource grid; manual is a fixed CV/test source."
}

if ($ParameterSource -eq "manual" -and -not $ParameterConfig) {
    throw "-ParameterSource manual requires -ParameterConfig."
}
if ($ModelSource -eq "provided_checkpoint" -and -not $Checkpoint) {
    throw "-ModelSource provided_checkpoint requires -Checkpoint."
}

$python = Join-Path $PSScriptRoot ".venv_3dino\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "3DINO Python environment not found: $python"
}

$pythonArguments = @(
    "Code\Models\ThreeDINO\train_full_finetune.py",
    "--stage", $Stage,
    "--percentage", $Percentage,
    "--parameter_source", $ParameterSource,
    "--model_source", $ModelSource,
    "--cache_namespace", $CacheNamespace,
    "--device", $Device,
    "--seed", $Seed,
    "--num_workers", $NumWorkers
)

if ($ParameterConfig) {
    $pythonArguments += @("--parameter_config", $ParameterConfig)
}
if ($GridConfig) {
    $pythonArguments += @("--grid_config", $GridConfig)
}
if ($MaxGridCandidates -gt 0) {
    $pythonArguments += @("--max_grid_candidates", $MaxGridCandidates)
}
if ($Checkpoint) {
    $pythonArguments += @("--checkpoint", $Checkpoint)
}
if ($ResultsRoot) {
    $pythonArguments += @("--results_root", $ResultsRoot)
}
if ($CacheRoot) {
    $pythonArguments += @("--cache_root", $CacheRoot)
}
if ($ModelRepoId) {
    $pythonArguments += @("--model_repo_id", $ModelRepoId)
}
if ($ModelRevision) {
    $pythonArguments += @("--model_revision", $ModelRevision)
}
if ($UpstreamRoot) {
    $pythonArguments += @("--upstream_root", $UpstreamRoot)
}
if ($UpstreamRevision) {
    $pythonArguments += @("--upstream_revision", $UpstreamRevision)
}
if ($PretrainedCheckpoint) {
    $pythonArguments += @("--pretrained_checkpoint", $PretrainedCheckpoint)
}
if ($PretrainedCheckpointSha256) {
    $pythonArguments += @("--pretrained_checkpoint_sha256", $PretrainedCheckpointSha256)
}
if ($ProvisionLog) {
    $pythonArguments += @("--provision_log", $ProvisionLog)
}
if ($LocalFilesOnly) {
    $pythonArguments += "--local_files_only"
}
if ($AllowExperimentalFullFineTune) {
    $pythonArguments += "--allow_experimental_full_finetune"
}
if (-not $NoResume) {
    $pythonArguments += "--resume"
}
if ($DryRun) {
    $pythonArguments += "--dry_run"
}
if ($ForceTest) {
    $pythonArguments += "--force_test"
}

Push-Location $PSScriptRoot
try {
    & $python @pythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "3DINO full-fine-tune command failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
