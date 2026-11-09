[CmdletBinding()]
param(
    [ValidateSet("plan", "access", "run")]
    [string]$Action = "plan",

    [string]$PromptProfile = "Code\Models\Pillar0\zero_shot_prompts_five_class_margin_v2.json",
    [string]$CacheNamespace = "pillar0_abdct_rave_1p25mm_384cube_rawhu_lz4_v1",
    [string]$Device = "cuda",
    [switch]$Force
)

$python = Join-Path $PSScriptRoot ".venv_pillar0\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Pillar-0 Python environment not found: $python"
}

$pythonArguments = @(
    "Code\Models\Pillar0\run_zero_shot.py",
    "--action", $Action,
    "--prompt_profile", $PromptProfile,
    "--cache_namespace", $CacheNamespace,
    "--device", $Device
)

if ($Force) {
    $pythonArguments += "--force"
}

Push-Location $PSScriptRoot
try {
    & $python @pythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Pillar-0 zero-shot command failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
