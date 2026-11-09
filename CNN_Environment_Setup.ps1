[CmdletBinding()]
param([string]$BootstrapPython = "", [switch]$DryRun)
$ErrorActionPreference = "Stop"
$environmentRoot = Join-Path $PSScriptRoot ".venv_cnn"
$environmentPython = Join-Path $environmentRoot "Scripts\python.exe"
function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    if ($DryRun) { Write-Host "DRY RUN: $Executable $($Arguments -join ' ')"; return }
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Executable" }
}
if (-not (Test-Path -LiteralPath $environmentPython)) {
    if ($BootstrapPython) {
        Invoke-Checked $BootstrapPython @("-m", "venv", $environmentRoot)
    } else {
        Invoke-Checked "py" @("-3.11", "-m", "venv", $environmentRoot)
    }
}
Invoke-Checked $environmentPython @("-c", "import sys; assert sys.version_info[:2] == (3,11), 'Requires Python 3.11'")
Invoke-Checked $environmentPython @("-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cu128", "torch==2.8.0")
Invoke-Checked $environmentPython @("-m", "pip", "install", "-r", (Join-Path $PSScriptRoot "Code\Models\_cnn\requirements.txt"))
Invoke-Checked $environmentPython @("-m", "pip", "check")
Invoke-Checked $environmentPython @("-c", "import torch, monai; print('torch',torch.__version__,'monai',monai.__version__,'CUDA',torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA unavailable'")
