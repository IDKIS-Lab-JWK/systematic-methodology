param(
    [ValidateSet("grid", "cv", "test", "all")]
    [string]$Stage = "all",

    [int[]]$Percentages = @(20, 40, 60, 80, 100),

    [ValidateSet("auto", "grid", "default", "publication")]
    [string]$ParameterSource = "publication",

    [ValidateSet("refit_full_development", "cv_ensemble", "best_cv_fold")]
    [string]$ModelSource = "cv_ensemble",

    [int]$NumWorkers = 4,
    [int]$FeatureExtractionBatchSize = 1,
    [string]$CacheRoot = "",
    [switch]$NoAmp,
    [switch]$NoResume,
    [switch]$ForceGrid,
    [switch]$ForceCV,
    [switch]$DryRun
)

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($Stage -eq "grid" -and $ParameterSource -notin @("auto", "grid")) {
    throw "Stage grid requires -ParameterSource auto or grid."
}
Push-Location $scriptRoot
try {
    foreach ($percentage in $Percentages) {
        $arguments = @(
            "Code\Models\Merlin\train_linear_probe.py",
            "--stage", $Stage,
            "--percentage", $percentage,
            "--parameter_source", $ParameterSource,
            "--model_source", $ModelSource,
            "--num_workers", $NumWorkers,
            "--feature_extraction_batch_size", $FeatureExtractionBatchSize
        )
        if (-not $NoAmp) { $arguments += "--amp" }
        if (-not $NoResume) { $arguments += "--resume" }
        if ($ForceGrid) { $arguments += "--force_grid" }
        if ($ForceCV) { $arguments += "--force_cv" }
        if ($DryRun) { $arguments += "--dry_run" }
        if ($CacheRoot) { $arguments += @("--cache_root", $CacheRoot) }

        python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Merlin linear probe failed for ${percentage}% (exit $LASTEXITCODE)."
        }
    }
}
finally {
    Pop-Location
}
