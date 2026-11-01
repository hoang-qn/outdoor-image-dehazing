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

$suiteDir = Join-Path $projectRoot "runs\dehazeformer_t_m1_m2_m3_full50"
$backgroundDir = Join-Path $suiteDir "background"
New-Item -ItemType Directory -Force -Path $backgroundDir | Out-Null
$suitePidFile = Join-Path $backgroundDir "suite.pid"
$suiteStatusFile = Join-Path $backgroundDir "suite_status.json"
$suiteStdout = Join-Path $backgroundDir "suite.stdout.log"
$suiteStderr = Join-Path $backgroundDir "suite.stderr.log"

if (Test-Path -LiteralPath $suitePidFile) {
    $existingPid = 0
    [void][int]::TryParse((Get-Content -LiteralPath $suitePidFile -Raw).Trim(), [ref]$existingPid)
    $suiteState = $null
    if (Test-Path -LiteralPath $suiteStatusFile) {
        $suiteState = (Get-Content -LiteralPath $suiteStatusFile -Raw | ConvertFrom-Json).status
    }
    if ($suiteState -in @("running", "waiting_for_pilot") -and $existingPid -gt 0 -and (Get-Process -Id $existingPid -ErrorAction SilentlyContinue)) {
        throw "The M1/M2/M3 full50 suite is already running (PID $existingPid). Check suite_status.json."
    }
    Remove-Item -LiteralPath $suitePidFile -Force
}

$workerScript = Join-Path $PSScriptRoot "run_m1_m2_m3_full50.py"
$pythonArgs = @(
    "-u", $workerScript,
    "--project-root", $projectRoot,
    "--python-executable", $pythonPath
)
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
$launchInfo = @{
    pid = $process.Id
    python = $pythonPath
    started_at = (Get-Date).ToString("o")
    epochs_per_model = 50
    early_stopping = $false
    models = @("M1", "M2", "M3")
}
$launchInfo | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $backgroundDir "suite_launch.json") -Encoding utf8

Write-Host "M1/M2/M3 full 50-epoch suite started in the background. PID: $($process.Id)"
Write-Host "It will wait for the current M1/M2/M3 pilot suite to finish before using the GPU."
Write-Host "Suite status: $suiteDir\background\suite_status.json"
Write-Host "Suite log: $suiteStdout"
Write-Host "Per-model logs/checkpoints: runs\dehazeformer_t_m1_full50, runs\dehazeformer_t_m2_full50, runs\dehazeformer_t_m3_full50"
