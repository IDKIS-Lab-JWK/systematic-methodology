[CmdletBinding()]
param(
    [ValidateSet("publication", "grid")]
    [string]$ParameterSource = "publication",

    [string]$GridConfig = "",

    [ValidateRange(0, 1000)]
    [int]$MaxGridCandidates = 0,

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

if ($ParameterSource -eq "publication" -and $GridConfig) {
    throw "-ParameterSource publication does not accept -GridConfig; the audited 20-candidate grid is fixed."
}
if ($ParameterSource -eq "publication" -and $MaxGridCandidates -gt 0) {
    throw "-ParameterSource publication does not accept -MaxGridCandidates; all 20 candidates are required."
}

$runner = Join-Path $PSScriptRoot "COLIPRI_Pooling_Optimised_Linear_Probe_Run_Script.ps1"
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "COLIPRI pooling-optimised probe launcher not found: $runner"
}

$percentages = @(80, 60, 40, 20)
$modelSources = @("refit_full_development", "cv_ensemble", "best_cv_fold")

foreach ($percentage in $percentages) {
    for ($sourceIndex = 0; $sourceIndex -lt $modelSources.Count; $sourceIndex++) {
        $modelSource = $modelSources[$sourceIndex]
        $stage = if ($sourceIndex -eq 0) { "all" } else { "test" }
        Write-Host "COLIPRI: ${percentage}% | stage=$stage | model_source=$modelSource"

        $arguments = @{
            Stage = $stage
            Percentage = $percentage
            ParameterSource = $ParameterSource
            ModelSource = $modelSource
            CacheNamespace = $CacheNamespace
            Device = $Device
            FeaturePrecision = $FeaturePrecision
            TrainingPrecision = $TrainingPrecision
            SourceFingerprint = $SourceFingerprint
            MicroBatchSize = $MicroBatchSize
            CheckpointInterval = $CheckpointInterval
            Seed = $Seed
            NumWorkers = $NumWorkers
        }
        if ($GridConfig) { $arguments.GridConfig = $GridConfig }
        if ($MaxGridCandidates -gt 0) { $arguments.MaxGridCandidates = $MaxGridCandidates }
        if ($DatasetRoot) { $arguments.DatasetRoot = $DatasetRoot }
        if ($ResultsRoot) { $arguments.ResultsRoot = $ResultsRoot }
        if ($CacheRoot) { $arguments.CacheRoot = $CacheRoot }
        if ($UpstreamRoot) { $arguments.UpstreamRoot = $UpstreamRoot }
        if ($UpstreamRevision) { $arguments.UpstreamRevision = $UpstreamRevision }
        if ($Checkpoint) { $arguments.Checkpoint = $Checkpoint }
        if ($ProvisionLog) { $arguments.ProvisionLog = $ProvisionLog }
        if ($LocalFilesOnly) { $arguments.LocalFilesOnly = $true }
        if ($NoResume) { $arguments.NoResume = $true }
        if ($NoPrefetch) { $arguments.NoPrefetch = $true }
        if ($DryRun) { $arguments.DryRun = $true }
        if ($ForceTest) { $arguments.ForceTest = $true }

        & $runner @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "COLIPRI wrapper stopped after a failed ${percentage}% $modelSource run (exit $LASTEXITCODE)."
        }
    }
}
