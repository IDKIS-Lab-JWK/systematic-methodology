[CmdletBinding()]
param(
    [ValidateSet("grid", "search", "cv", "test", "all")]
    [string]$Stage = "all",

    [ValidateSet(20, 40, 60, 80, 100)]
    [int]$Percentage = 40,

    [ValidateSet("default", "publication", "search", "manual")]
    [string]$ParameterSource = "default",

    [string]$ParameterConfig = "",
    [string]$GridConfig = "Code\Models\Pillar0\search_linear_probe.json",

    [ValidateRange(0, 100)]
    [int]$MaxGridCandidates = 0,

    [ValidateSet("refit_full_development", "cv_ensemble", "best_cv_fold", "provided_checkpoint")]
    [string]$ModelSource = "best_cv_fold",

    [string]$Checkpoint = "",
    [string]$CacheNamespace = "pillar0_abdct_rave_1p25mm_384cube_rawhu_lz4_v1",
    [string]$Device = "cuda",

    [ValidateRange(0, 64)]
    [int]$NumWorkers = 0,

    [switch]$Resume,
    [switch]$DryRun,
    [switch]$ForceTest
)

if ($ParameterSource -eq "manual" -and -not $ParameterConfig) {
    throw "-ParameterSource manual requires -ParameterConfig."
}
if ($ModelSource -eq "provided_checkpoint" -and -not $Checkpoint) {
    throw "-ModelSource provided_checkpoint requires -Checkpoint."
}

$python = Join-Path $PSScriptRoot ".venv_pillar0\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Pillar-0 Python environment not found: $python"
}

$pythonArguments = @(
    "Code\Models\Pillar0\train_linear_probe.py",
    "--stage", $Stage,
    "--percentage", $Percentage,
    "--parameter_source", $ParameterSource,
    "--cache_namespace", $CacheNamespace,
    "--model_source", $ModelSource,
    "--device", $Device,
    "--num_workers", $NumWorkers
)

if ($Stage -in @("grid", "search")) {
    $pythonArguments += @("--grid_config", $GridConfig)
}
if ($ParameterConfig) {
    $pythonArguments += @("--parameter_config", $ParameterConfig)
}
if ($MaxGridCandidates -gt 0) {
    $pythonArguments += @("--max_grid_candidates", $MaxGridCandidates)
}
if ($Checkpoint) {
    $pythonArguments += @("--checkpoint", $Checkpoint)
}
if ($Resume) {
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
        throw "Pillar-0 linear-probe command failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
