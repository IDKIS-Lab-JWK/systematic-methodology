[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet("MedicalNet", "DenseNet")][string]$Model,
    [ValidateSet("grid", "cv", "test", "all")][string]$Stage = "all",
    [ValidateSet(20,40,60,80,100)][int[]]$Percentages = @(100),
    [ValidateSet("publication", "grid", "default")][string]$ParameterSource = "publication",
    [ValidateSet("refit_full_development", "cv_ensemble", "best_cv_fold")][string]$ModelSource = "refit_full_development",
    [string]$PythonExe = "", [string]$GridConfig = "", [string]$ParameterConfig = "",
    [string]$Checkpoint = "", [string]$CacheRoot = "", [string]$ResultsRoot = "",
    [int]$MicroBatchSize = 4,
    [ValidateSet("auto", "bf16", "fp32")][string]$TrainingPrecision = "auto",
    [ValidateSet("temporary", "retain")][string]$CachePolicy = "temporary",
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
if (-not $PythonExe) { $PythonExe = Join-Path $PSScriptRoot ".venv_cnn\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $PythonExe)) { throw "Run CNN_Environment_Setup.ps1 first or pass -PythonExe." }
foreach ($percentage in $Percentages) {
    $arguments = @((Join-Path $PSScriptRoot "Code\Models\$Model\train.py"), "--stage", $Stage,
        "--percentage", "$percentage", "--parameter_source", $ParameterSource, "--model_source", $ModelSource,
        "--micro_batch_size", "$MicroBatchSize", "--training_precision", $TrainingPrecision, "--cache_policy", $CachePolicy)
    if ($GridConfig) { $arguments += @("--grid_config", $GridConfig) }
    if ($ParameterConfig) { $arguments += @("--parameter_config", $ParameterConfig) }
    if ($Checkpoint) { $arguments += @("--checkpoint", $Checkpoint) }
    if ($CacheRoot) { $arguments += @("--cache_root", $CacheRoot) }
    if ($ResultsRoot) { $arguments += @("--results_root", $ResultsRoot) }
    if ($DryRun) { $arguments += "--dry_run" }
    & $PythonExe @arguments
    if ($LASTEXITCODE -ne 0) { throw "$Model failed at $percentage percent (exit $LASTEXITCODE)." }
}
