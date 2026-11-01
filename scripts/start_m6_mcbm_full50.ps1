param([switch]$Resume)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project venv not found: $python"
}
$background = Join-Path $projectRoot "runs\dehazeformer_t_m6_mcbm_full50\background"
New-Item -ItemType Directory -Force -Path $background | Out-Null
$pidPath = Join-Path $background "pipeline.pid"
foreach ($activePidFile in @($pidPath, (Join-Path $background "train.pid"))) {
if (Test-Path -LiteralPath $activePidFile) {
    $oldPid = 0
    [void][int]::TryParse((Get-Content -LiteralPath $activePidFile -Raw).Trim(), [ref]$oldPid)
    if ($oldPid -gt 0 -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)) {
        throw "M6 + MCBM pipeline is already active (PID $oldPid)."
    }
}
}
$checkpoint = Join-Path $projectRoot "runs\dehazeformer_t_m6_mcbm_full50\checkpoints\last.pth"
if ((Test-Path -LiteralPath $checkpoint) -and -not $Resume) {
    throw "Existing run found; use -Resume to continue without overwriting it."
}
if ($Resume -and -not (Test-Path -LiteralPath $checkpoint)) {
    throw "No checkpoint is available to resume."
}
$stopPath = Join-Path $background "train.stop"
if (Test-Path -LiteralPath $stopPath) { Remove-Item -LiteralPath $stopPath -Force }
$worker = Join-Path $PSScriptRoot "run_m6_mcbm_full50.py"
$argumentLine = '-u "' + $worker + '"'
if ($Resume) { $argumentLine += ' --resume' }
$oldEncoding = $env:PYTHONIOENCODING
$env:PYTHONIOENCODING = "utf-8"
try {
    $process = Start-Process -FilePath $python -ArgumentList $argumentLine `
        -WorkingDirectory $projectRoot -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $background "pipeline.stdout.log") `
        -RedirectStandardError (Join-Path $background "pipeline.stderr.log") -PassThru
} finally {
    $env:PYTHONIOENCODING = $oldEncoding
}
Set-Content -LiteralPath $pidPath -Value $process.Id -Encoding ascii
Write-Host "M6 + MCBM pipeline started independently of this terminal (PID $($process.Id))."
Write-Host "Stages: missing depth maps -> lossless cache -> batch/worker tuning -> 50 training epochs."
Write-Host "Status: $background\pipeline_status.json"
Write-Host "Preparation/tuning log: $background\pipeline.stdout.log"
Write-Host "Training log: $background\train.stdout.log"
