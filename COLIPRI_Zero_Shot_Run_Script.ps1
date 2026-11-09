[CmdletBinding()]
param(
    [string]$DatasetRoot = "",
    [string]$ResultsRoot = "",
    [string]$PromptConfig = "",
    [string]$UpstreamRoot = "",
    [string]$Checkpoint = "",
    [string]$ProvisionLog = "",
    [string]$Device = "cuda",
    [ValidateSet("auto", "bf16", "fp16", "fp32")]
    [string]$Precision = "auto",
    [switch]$LocalFilesOnly,
    [switch]$NoResume,
    [switch]$DryRun
)

$python = Join-Path $PSScriptRoot ".venv_colipri\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "COLIPRI Python environment not found: $python"
}
$arguments = @(
    "Code\Models\COLIPRI\run_zero_shot.py",
    "--device", $Device,
    "--precision", $Precision
)
if ($DatasetRoot) { $arguments += @("--dataset_root", $DatasetRoot) }
if ($ResultsRoot) { $arguments += @("--results_root", $ResultsRoot) }
if ($PromptConfig) { $arguments += @("--prompt_config", $PromptConfig) }
if ($UpstreamRoot) { $arguments += @("--upstream_root", $UpstreamRoot) }
if ($Checkpoint) { $arguments += @("--checkpoint", $Checkpoint) }
if ($ProvisionLog) { $arguments += @("--provision_log", $ProvisionLog) }
if ($LocalFilesOnly) { $arguments += "--local_files_only" }
if ($NoResume) { $arguments += "--no-resume" }
if ($DryRun) { $arguments += "--dry_run" }

Push-Location $PSScriptRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "COLIPRI zero-shot evaluation failed with exit code $LASTEXITCODE." }
}
finally { Pop-Location }
