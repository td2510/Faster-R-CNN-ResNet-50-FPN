param(
    [string]$Config = "track_c_config.json"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $repo ".venv-track-c\Scripts\python.exe"
$runner = Join-Path $repo "track_c_faster_rcnn.py"
$artifacts = Join-Path $repo "data\BTL_DeTai4_9000\runs_rcnn"
$incoming = Join-Path $repo "incoming"
$roboflowYaml = Join-Path $repo "data\BTL_DeTai4\roboflow_v5\data.yaml"
$apiKey = Join-Path $repo "incoming\roboflow_api_key.txt"
$subsetYaml = Join-Path $repo "data\BTL_DeTai4\dataset_subset\data.yaml"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing CUDA environment: $python"
}

if (-not (Test-Path -LiteralPath $subsetYaml)) {
    if (-not (Test-Path -LiteralPath $roboflowYaml)) {
        if (Test-Path -LiteralPath $apiKey) {
            & $python $runner roboflow --config $Config
            if ($LASTEXITCODE -ne 0) { throw "Roboflow download failed" }
        } else {
            $archives = Get-ChildItem -LiteralPath $incoming -Filter "*.zip" -ErrorAction SilentlyContinue
            if (-not $archives) {
                throw "Create $apiKey with one private Roboflow API key, or place Drive ZIP file(s) in $incoming"
            }
            & $python $runner extract --config $Config
            if ($LASTEXITCODE -ne 0) { throw "ZIP extraction failed" }
        }
    }
    & $python $runner subset --config $Config
    if ($LASTEXITCODE -ne 0) { throw "Subset creation failed" }
}

New-Item -ItemType Directory -Force -Path $artifacts | Out-Null
$stdout = Join-Path $artifacts "track_c.stdout.log"
$stderr = Join-Path $artifacts "track_c.stderr.log"
$process = Start-Process -FilePath $python `
    -ArgumentList @("-u", $runner, "all", "--config", $Config) `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$process.Id | Set-Content -LiteralPath (Join-Path $artifacts "track_c.pid") -Encoding ascii
Write-Host "Track C started with PID $($process.Id)"
Write-Host "stdout: $stdout"
Write-Host "stderr: $stderr"
