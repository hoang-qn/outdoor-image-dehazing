param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$RunName = "dehazeformer_t"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$backgroundDir = Join-Path $projectRoot "runs\$RunName\background"
$pidFile = Join-Path $backgroundDir "train.pid"
$stopFile = Join-Path $backgroundDir "train.stop"
if (-not (Test-Path -LiteralPath $pidFile)) {
    throw "No PID file found for run '$RunName'."
}

$trainingPid = 0
[void][int]::TryParse((Get-Content -LiteralPath $pidFile -Raw).Trim(), [ref]$trainingPid)
if ($trainingPid -le 0) {
    throw "PID file is invalid: $pidFile"
}
$processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $trainingPid" -ErrorAction SilentlyContinue
if (-not $processInfo -or $processInfo.CommandLine -notmatch "dehazing\.train") {
    Remove-Item -LiteralPath $pidFile -Force
    Write-Host "Training process is not running; removed stale PID file."
    exit 0
}

Set-Content -LiteralPath $stopFile -Value "Stop after current epoch: $(Get-Date -Format o)" -Encoding utf8
Write-Host "Requested a stop after the current epoch (PID $trainingPid). The current epoch will finish and checkpoints will be saved."
