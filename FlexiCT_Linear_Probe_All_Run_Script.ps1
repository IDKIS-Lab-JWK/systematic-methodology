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
    [string]$CacheNamespace = "flexict_vlm_lps_2mm_160cube_clsmean_proj_l2_v1",
    [string]$UpstreamRoot = "",
    [string]$UpstreamRevision = "",
    [string]$Checkpoint = "",
    [string]$CheckpointSha256 = "",
    [string]$ProvisionLog = "",
    [string]$Device = "cuda",

    [ValidateSet("auto", "bf16", "fp16", "fp32")]
    [string]$FeaturePrecision = "auto",

    [ValidateSet("stat", "sha256")]
    [string]$SourceFingerprint = "sha256",

    [int]$Seed = 42,

    [ValidateSet(0)]
    [int]$NumWorkers = 0,

    [switch]$LocalFilesOnly,
    [switch]$AllowTrustedLegacyCheckpoint,
    [switch]$NoResume,
    [switch]$DryRun,
    [switch]$ForceTest
)

if ($AllowTrustedLegacyCheckpoint -and -not $CheckpointSha256) {
    throw "-AllowTrustedLegacyCheckpoint requires -CheckpointSha256."
}
if (-not $DryRun -and $SourceFingerprint -ne "sha256") {
    throw "FlexiCT linear-probe execution requires -SourceFingerprint sha256; stat is cache diagnostics/dry-run only."
}
if ($ParameterSource -eq "publication" -and $GridConfig) {
    throw "-ParameterSource publication does not accept -GridConfig; the audited seven-C grid is fixed."
}
if ($ParameterSource -eq "publication" -and $MaxGridCandidates -gt 0) {
    throw "-ParameterSource publication does not accept -MaxGridCandidates; all seven candidates are required."
}

$runner = Join-Path $PSScriptRoot "FlexiCT_Linear_Probe_Run_Script.ps1"
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "FlexiCT linear-probe launcher not found: $runner"
}

$percentages = @(100, 80, 60, 40, 20)
$modelSources = @("refit_full_development", "cv_ensemble", "best_cv_fold")

foreach ($percentage in $percentages) {
    for ($sourceIndex = 0; $sourceIndex -lt $modelSources.Count; $sourceIndex++) {
        $modelSource = $modelSources[$sourceIndex]
        $stage = if ($sourceIndex -eq 0) { "all" } else { "test" }
        Write-Host "FlexiCT: ${percentage}% | stage=$stage | model_source=$modelSource"

        $arguments = @{
            Stage = $stage
            Percentage = $percentage
            ParameterSource = $ParameterSource
            ModelSource = $modelSource
            CacheNamespace = $CacheNamespace
            Device = $Device
            FeaturePrecision = $FeaturePrecision
            SourceFingerprint = $SourceFingerprint
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
        if ($CheckpointSha256) { $arguments.CheckpointSha256 = $CheckpointSha256 }
        if ($ProvisionLog) { $arguments.ProvisionLog = $ProvisionLog }
        if ($LocalFilesOnly) { $arguments.LocalFilesOnly = $true }
        if ($AllowTrustedLegacyCheckpoint) { $arguments.AllowTrustedLegacyCheckpoint = $true }
        if ($NoResume) { $arguments.NoResume = $true }
        if ($DryRun) { $arguments.DryRun = $true }
        if ($ForceTest) { $arguments.ForceTest = $true }

        & $runner @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "FlexiCT wrapper stopped after a failed ${percentage}% $modelSource run (exit $LASTEXITCODE)."
        }
    }
}
