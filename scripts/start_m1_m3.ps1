param(
    [string]$PythonExecutable = "python",
    [int]$Epochs = 12
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

$suiteDir = Join-Path $projectRoot "runs\dehazeformer_t_m1_m3_pilot12"
$backgroundDir = Join-Path $suiteDir "background"
New-Item -ItemType Directory -Force -Path $backgroundDir | Out-Null
$suitePidFile = Join-Path $backgroundDir "suite.pid"
$suiteStopFile = Join-Path $backgroundDir "stop_after_current_model"
$suiteStdout = Join-Path $backgroundDir "suite.stdout.log"
$suiteStderr = Join-Path $backgroundDir "suite.stderr.log"

if (Test-Path -LiteralPath $suitePidFile) {
    $existingPid = 0
    [void][int]::TryParse((Get-Content -LiteralPath $suitePidFile -Raw).Trim(), [ref]$existingPid)
    if ($existingPid -gt 0 -and (Get-Process -Id $existingPid -ErrorAction SilentlyContinue)) {
        throw "The M1/M2/M3 suite is already running (PID $existingPid). Check suite_status.json."
    }
    Remove-Item -LiteralPath $suitePidFile -Force
}

$configs = @(
    "configs\experiment\dehazeformer_t_m1_pilot.yaml",
    "configs\experiment\dehazeformer_t_m2_pilot.yaml",
    "configs\experiment\dehazeformer_t_m3_pilot.yaml"
)
foreach ($relativeConfig in $configs) {
    $configPath = Join-Path $projectRoot $relativeConfig
    if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
        throw "Experiment config not found: $configPath"
    }
}

if (Test-Path -LiteralPath $suiteStopFile) {
    Remove-Item -LiteralPath $suiteStopFile -Force
}
$workerScript = Join-Path $PSScriptRoot "run_model_ablations.py"
$pythonArgs = @(
    "-u", $workerScript,
    "--project-root", $projectRoot,
    "--python-executable", $pythonPath
)
if ($Epochs -gt 0) {
    $pythonArgs += @("--epochs", "$Epochs")
}
$argumentLine = ($pythonArgs | ForEach-Object { '"' + ([string]$_).Replace('"', '\"') + '"' }) -join " "

$oldPythonPath = $env:PYTHONPATH
$srcPath = Join-Path $projectRoot "src"
$env:PYTHONPATH = if ($oldPythonPath) { "$srcPath;$oldPythonPath" } else { $srcPath }
try {
    $process = Start-Process `
        -FilePath $pythonPath `
        -ArgumentList $argumentLine `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $suiteStdout `
        -RedirectStandardError $suiteStderr `
        -PassThru
}
finally {
    $env:PYTHONPATH = $oldPythonPath
}

Set-Content -LiteralPath $suitePidFile -Value $process.Id -Encoding ascii
$epochsPerModel = if ($Epochs -gt 0) { $Epochs } else { 12 }
$launchInfo = @{
    pid = $process.Id
    python = $pythonPath
    started_at = (Get-Date).ToString("o")
    epochs_per_model = $epochsPerModel
    models = @("M1", "M2", "M3")
}
$launchInfo | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $backgroundDir "suite_launch.json") -Encoding utf8

Write-Host "M1/M2/M3 pilot suite started in the background. PID: $($process.Id)"
Write-Host "Suite status: $suiteDir\background\suite_status.json"
Write-Host "Suite log: $suiteStdout"
Write-Host "Per-model logs/checkpoints: runs\dehazeformer_t_m1_pilot12, runs\dehazeformer_t_m2_pilot12, runs\dehazeformer_t_m3_pilot12"
