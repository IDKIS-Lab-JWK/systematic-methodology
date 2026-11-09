param(
    [ValidateSet("both", "five_class_adapted", "author_binary")]
    [string]$Mode = "both",
    [int]$BatchSize = 1,
    [int]$NumWorkers = 0,
    [string]$CacheRoot = "",
    [switch]$Cpu,
    [switch]$NoAmp,
    [switch]$DryRun,
    [switch]$ValidateSetup,
    [switch]$SkipCheckpointHash,
    [switch]$Force
)

$pythonArguments = @(
    "Code\Models\Merlin\run_zero_shot.py",
    "--mode", $Mode,
    "--batch_size", $BatchSize,
    "--num_workers", $NumWorkers
)

if ($CacheRoot) {
    $pythonArguments += @("--cache_root", $CacheRoot)
}
if ($Cpu) {
    $pythonArguments += @("--device", "cpu")
}
elseif (-not $NoAmp -and -not $DryRun) {
    $pythonArguments += "--amp"
}
if ($DryRun) {
    $pythonArguments += "--dry_run"
}
if ($ValidateSetup) {
    $pythonArguments += "--validate_setup"
}
if ($SkipCheckpointHash) {
    $pythonArguments += "--skip_checkpoint_hash"
}
if ($Force) {
    $pythonArguments += "--force"
}

Push-Location $PSScriptRoot
try {
    & python @pythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Merlin zero-shot run failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
