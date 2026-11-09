<#
.SYNOPSIS
Plan, prepare, or validate COLIPRI's permanent probe and prepared-CT caches.
.DESCRIPTION
CacheTarget finetune builds CT inputs shared by both fine-tuning scopes.
The training launcher creates temporary frozen outputs automatically when
InputSource is frozen (the training default), and removes them after the
whole requested launch succeeds. This cache script does not build or delete
those temporary outputs. Action defaults to plan; CacheTarget defaults to both.
#>
[CmdletBinding()]
param(
    [ValidateSet("plan", "prepare", "validate")]
    [string]$Action = "plan",

    [ValidateSet("probe", "finetune", "both")]
    [string]$CacheTarget = "both",

    [ValidateSet(100)]
    [int]$Percentage = 100,

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

    [ValidateSet("stat", "sha256")]
    [string]$SourceFingerprint = "sha256",

    [ValidateSet(0)]
    [int]$NumWorkers = 0,

    [switch]$LocalFilesOnly,
    [switch]$SkipArtifactValidation,
    [switch]$DryRun
)

if ($SkipArtifactValidation -and $Action -ne "validate") {
    throw "-SkipArtifactValidation is valid only with -Action validate."
}

$python = Join-Path $PSScriptRoot ".venv_colipri\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "COLIPRI Python environment not found: $python. Run COLIPRI_Environment_Setup.ps1 first."
}

$pythonArguments = @(
    "Code\Models\COLIPRI\prepare_cache.py",
    "--action", $Action,
    "--cache_target", $CacheTarget,
    "--percentage", $Percentage,
    "--cache_namespace", $CacheNamespace,
    "--device", $Device,
    "--feature_precision", $FeaturePrecision,
    "--source_fingerprint", $SourceFingerprint,
    "--num_workers", $NumWorkers
)
if ($DatasetRoot) { $pythonArguments += @("--dataset_root", $DatasetRoot) }
if ($ResultsRoot) { $pythonArguments += @("--results_root", $ResultsRoot) }
if ($CacheRoot) { $pythonArguments += @("--cache_root", $CacheRoot) }
if ($UpstreamRoot) { $pythonArguments += @("--upstream_root", $UpstreamRoot) }
if ($UpstreamRevision) { $pythonArguments += @("--upstream_revision", $UpstreamRevision) }
if ($Checkpoint) { $pythonArguments += @("--checkpoint", $Checkpoint) }
if ($ProvisionLog) { $pythonArguments += @("--provision_log", $ProvisionLog) }
if ($LocalFilesOnly) { $pythonArguments += "--local_files_only" }
if ($SkipArtifactValidation) { $pythonArguments += "--skip_artifact_validation" }
if ($DryRun) { $pythonArguments += "--dry_run" }

Push-Location $PSScriptRoot
try {
    & $python @pythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "COLIPRI cache command failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
