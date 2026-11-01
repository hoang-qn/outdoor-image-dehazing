param(
    [string]$PythonExecutable = "python",
    [string]$Config = "configs/experiment/baseline.yaml",
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$RunName = "dehazeformer_t",
    [int]$Epochs = 0,
    [int]$BatchSize = 0,
    [int]$CropSize = 0,
    [string]$ResumeFrom,
    [switch]$NoAmp
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$configPath = if ([IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $projectRoot $Config }
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "Config file not found: $configPath"
}

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

$runDir = Join-Path $projectRoot "runs\$RunName"
$backgroundDir = Join-Path $runDir "background"
New-Item -ItemType Directory -Force -Path $backgroundDir | Out-Null
$pidFile = Join-Path $backgroundDir "train.pid"
$stopFile = Join-Path $backgroundDir "train.stop"
$stdoutLog = Join-Path $backgroundDir "train.stdout.log"
$stderrLog = Join-Path $backgroundDir "train.stderr.log"

if (Test-Path -LiteralPath $pidFile) {
    $existingPid = 0
    [void][int]::TryParse((Get-Content -LiteralPath $pidFile -Raw).Trim(), [ref]$existingPid)
    if ($existingPid -gt 0) {
        $existingProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
        if ($existingProcess -and $existingProcess.CommandLine -match "dehazing\.train") {
            throw "Training is already running (PID $existingPid). Check it with scripts\training_status.ps1."
        }
    }
    Remove-Item -LiteralPath $pidFile -Force
}

$pythonArgs = @("-m", "dehazing.train", "--config", $configPath, "--output-dir", $runDir)
$pythonArgs += @("--stop-file", $stopFile)
if ($ResumeFrom) {
    $resumePath = if ([IO.Path]::IsPathRooted($ResumeFrom)) { $ResumeFrom } else { Join-Path $projectRoot $ResumeFrom }
    if (-not (Test-Path -LiteralPath $resumePath -PathType Leaf)) {
        throw "Resume checkpoint not found: $resumePath"
    }
    $resumePath = (Resolve-Path -LiteralPath $resumePath).Path
    $pythonArgs += @("--resume", $resumePath)
    if ([IO.Path]::GetFileName($resumePath) -eq "best.pth") {
        $checkpointDir = Join-Path $runDir "checkpoints"
        New-Item -ItemType Directory -Force -Path $checkpointDir | Out-Null
        $runBestCheckpoint = Join-Path $checkpointDir "best.pth"
        if (-not (Test-Path -LiteralPath $runBestCheckpoint)) {
            Copy-Item -LiteralPath $resumePath -Destination $runBestCheckpoint
        }
    }
}
if ($Epochs -gt 0) { $pythonArgs += @("--epochs", "$Epochs") }
if ($BatchSize -gt 0) { $pythonArgs += @("--batch-size", "$BatchSize") }
if ($CropSize -gt 0) { $pythonArgs += @("--crop-size", "$CropSize") }
if ($NoAmp) { $pythonArgs += "--no-amp" }

# Start-Process accepts a command-line string; quote each argument for paths with spaces.
$argumentLine = ($pythonArgs | ForEach-Object { '"' + ([string]$_).Replace('"', '\"') + '"' }) -join " "
if (Test-Path -LiteralPath $stopFile) { Remove-Item -LiteralPath $stopFile -Force }
$oldPythonPath = $env:PYTHONPATH
$srcPath = Join-Path $projectRoot "src"
$env:PYTHONPATH = if ($oldPythonPath) { "$srcPath;$oldPythonPath" } else { $srcPath }
try {
    $process = Start-Process `
        -FilePath $pythonPath `
        -ArgumentList $argumentLine `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -PassThru
}
finally {
    $env:PYTHONPATH = $oldPythonPath
}

Set-Content -LiteralPath $pidFile -Value $process.Id -Encoding ascii
$launchInfo = @{
    pid = $process.Id
    python = $pythonPath
    config = $configPath
    run_dir = $runDir
    started_at = (Get-Date).ToString("o")
    arguments = $pythonArgs
}
$launchInfo | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $backgroundDir "launch.json") -Encoding utf8

Write-Host "Training started in the background. PID: $($process.Id)"
Write-Host "Run directory: $runDir"
Write-Host "Check status: .\scripts\training_status.ps1 -RunName $RunName"
Write-Host "Stop training: .\scripts\stop_training.ps1 -RunName $RunName"
