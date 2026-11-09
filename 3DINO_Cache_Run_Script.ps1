[CmdletBinding()]
param(
    [ValidateSet("plan", "prepare", "validate")]
    [string]$Action = "plan",

    [ValidateSet(20, 40, 60, 80, 100)]
    [int]$Percentage = 100,

    [string]$DatasetRoot = "",
    [string]$SplitManifest = "",
    [string]$CacheRoot = "",
    [string]$CacheNamespace = "threedino_ct_ras_p0005_p9995_fg_resize_144x144x112_v2",
    [string]$Output = "",

    [switch]$DryRun,
    [switch]$SkipArtifactValidation,
    [switch]$Quiet
)

$python = Join-Path $PSScriptRoot ".venv_3dino\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "3DINO Python environment not found: $python"
}

$pythonArguments = @(
    "Code\Models\ThreeDINO\prepare_cache.py",
    "--action", $Action,
    "--percentage", $Percentage,
    "--cache_namespace", $CacheNamespace
)

if ($DatasetRoot) {
    $pythonArguments += @("--dataset_root", $DatasetRoot)
}
if ($SplitManifest) {
    $pythonArguments += @("--split_manifest", $SplitManifest)
}
if ($CacheRoot) {
    $pythonArguments += @("--cache_root", $CacheRoot)
}
if ($Output) {
    $pythonArguments += @("--output", $Output)
}
if ($DryRun) {
    $pythonArguments += "--dry_run"
}
if ($SkipArtifactValidation) {
    $pythonArguments += "--skip_artifact_validation"
}
if ($Quiet) {
    $pythonArguments += "--quiet"
}

Push-Location $PSScriptRoot
try {
    & $python @pythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "3DINO cache command failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
