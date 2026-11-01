param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$RunName = "dehazeformer_t",
    [int]$Tail = 20
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$backgroundDir = Join-Path $projectRoot "runs\$RunName\background"
$pidFile = Join-Path $backgroundDir "train.pid"
$stdoutLog = Join-Path $backgroundDir "train.stdout.log"
$stderrLog = Join-Path $backgroundDir "train.stderr.log"

if (-not (Test-Path -LiteralPath $pidFile)) {
    Write-Host "No PID file found for run '$RunName'."
    exit 1
}

$trainingPid = 0
[void][int]::TryParse((Get-Content -LiteralPath $pidFile -Raw).Trim(), [ref]$trainingPid)
$processInfo = if ($trainingPid -gt 0) {
    Get-CimInstance Win32_Process -Filter "ProcessId = $trainingPid" -ErrorAction SilentlyContinue
}

if ($processInfo -and $processInfo.CommandLine -match "dehazing\.train") {
    Write-Host "Training is running (PID $trainingPid)."
}
else {
    Write-Host "Training process is not running. The PID file may be stale."
}

if (Test-Path -LiteralPath $stdoutLog) {
    Write-Host "`n--- Recent standard output ---"
    Get-Content -LiteralPath $stdoutLog -Tail $Tail
}
if (Test-Path -LiteralPath $stderrLog) {
    Write-Host "`n--- Recent errors ---"
    Get-Content -LiteralPath $stderrLog -Tail $Tail
}
