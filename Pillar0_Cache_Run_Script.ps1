<#
Prepares the permanent raw-HU RAVE cache. Atlas partial fine-tuning creates
all ten rotation-specific frozen representations automatically at launch;
those temporary representations are removed only after a successful run.
#>
[CmdletBinding()]
param(
    [ValidateSet("plan", "prepare", "validate")]
    [string]$Action = "plan",

    [ValidateSet(20, 40, 60, 80, 100)]
    [int]$Percentage = 100,

    [string]$CacheNamespace = "pillar0_abdct_rave_1p25mm_384cube_rawhu_lz4_v1",

    [ValidateSet("stat", "sha256")]
    [string]$SourceFingerprint = "stat",

    [ValidateRange(1, 128)]
    [int]$Workers = 4,

    [ValidateRange(0, 100)]
    [int]$SmokePatientsPerClass = 0,

    [string]$Output = "",
    [switch]$DryRun,
    [switch]$SkipVolumeValidation
)

$python = Join-Path $PSScriptRoot ".venv_pillar0\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Pillar-0 Python environment not found: $python"
}

$pythonArguments = @(
    "Code\Models\Pillar0\prepare_cache.py",
    "--action", $Action,
    "--percentage", $Percentage,
    "--cache_namespace", $CacheNamespace,
    "--config", "Code\Models\Pillar0\rave_abdomen_384_lz4.yaml",
    "--source_fingerprint", $SourceFingerprint,
    "--workers", $Workers,
    "--expected_shape_dhw", 384, 384, 384,
    "--expected_spacing_xyz_mm", 1.25, 1.25, 1.25
)

if ($SmokePatientsPerClass -gt 0) {
    $pythonArguments += @("--smoke_patients_per_class", $SmokePatientsPerClass)
}
if ($Output) {
    $pythonArguments += @("--output", $Output)
}
if ($DryRun) {
    $pythonArguments += "--dry_run"
}
if ($SkipVolumeValidation) {
    $pythonArguments += "--skip_volume_validation"
}

Push-Location $PSScriptRoot
try {
    & $python @pythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Pillar-0 cache command failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
