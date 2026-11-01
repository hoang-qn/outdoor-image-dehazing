param(
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = $null
if ($PythonExecutable -eq "python") {
    $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        $pythonPath = $venvPython
    }
}
if (-not $pythonPath) {
    $pythonCommand = Get-Command $PythonExecutable -ErrorAction Stop
    $pythonPath = $pythonCommand.Source
}

$suiteDir = Join-Path $projectRoot "runs\recent_lightweight_full50_bs8"
$backgroundDir = Join-Path $suiteDir "background"
New-Item -ItemType Directory -Force -Path $backgroundDir | Out-Null
$suitePidFile = Join-Path $backgroundDir "suite.pid"
$statusFile = Join-Path $backgroundDir "suite_status.json"
if (Test-Path -LiteralPath $suitePidFile) {
    $existingPid = 0
    [void][int]::TryParse((Get-Content -LiteralPath $suitePidFile -Raw).Trim(), [ref]$existingPid)
    if ($existingPid -gt 0 -and (Get-Process -Id $existingPid -ErrorAction SilentlyContinue)) {
        throw "The lightweight 50-epoch suite is already running (PID $existingPid). Check $statusFile."
    }
    Remove-Item -LiteralPath $suitePidFile -Force
}

$activeTraining = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "dehazing\.train|run_m1_m2_m3_full50|run_m6_mcbm_full50|run_recent_lightweight_full50" }
if ($activeTraining) {
    $details = ($activeTraining | ForEach-Object { "PID $($_.ProcessId): $($_.CommandLine)" }) -join "`n"
    throw "Another training process is active. Wait for it to finish before starting this GPU queue:`n$details"
}

$worker = Join-Path $PSScriptRoot "run_recent_lightweight_full50.py"
$pythonArgs = @("-u", $worker, "--project-root", $projectRoot, "--python-executable", $pythonPath)
$argumentLine = ($pythonArgs | ForEach-Object { '"' + ([string]$_).Replace('"', '\"') + '"' }) -join " "
$suiteStdout = Join-Path $backgroundDir "suite.stdout.log"
$suiteStderr = Join-Path $backgroundDir "suite.stderr.log"
$oldPythonPath = $env:PYTHONPATH
$srcPath = Join-Path $projectRoot "src"
$env:PYTHONPATH = if ($oldPythonPath) { "$srcPath;$oldPythonPath" } else { $srcPath }
try {
    $process = Start-Process -FilePath $pythonPath -ArgumentList $argumentLine `
        -WorkingDirectory $projectRoot -WindowStyle Hidden `
        -RedirectStandardOutput $suiteStdout -RedirectStandardError $suiteStderr -PassThru
}
finally {
    $env:PYTHONPATH = $oldPythonPath
}

Set-Content -LiteralPath $suitePidFile -Value $process.Id -Encoding ascii
$launchInfo = @{
    pid = $process.Id
    python = $pythonPath
    started_at = (Get-Date).ToString("o")
    epochs_per_model = 50
    batch_size = 8
    early_stopping = $false
    models = @("SLRNet", "WLD-Net")
    order = @("SLRNet", "WLD-Net")
}
$launchInfo | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $backgroundDir "suite_launch.json") -Encoding utf8

Write-Host "SLRNet and WLD-Net 50-epoch suite started in the background (batch size 8). PID: $($process.Id)"
Write-Host "Runs sequentially to avoid GPU contention."
Write-Host "Status: runs\recent_lightweight_full50_bs8\background\suite_status.json"
Write-Host "Per-model logs/checkpoints: runs\slrnet_recent_full50 and runs\wldnet_recent_full50"
