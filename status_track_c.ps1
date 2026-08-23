$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$artifacts = Join-Path $repo "data\BTL_DeTai4_9000\runs_rcnn"
$pidPath = Join-Path $artifacts "track_c.pid"
$heartbeatPath = Join-Path $artifacts "heartbeat.json"
$stdout = Join-Path $artifacts "track_c.stdout.log"
$stderr = Join-Path $artifacts "track_c.stderr.log"

if (Test-Path -LiteralPath $pidPath) {
    $trackPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    $process = Get-Process -Id $trackPid -ErrorAction SilentlyContinue
    if ($process) {
        Write-Host "Process: RUNNING (PID $trackPid, CPU $([math]::Round($process.CPU, 1))s)"
    } else {
        Write-Host "Process: EXITED (last PID $trackPid)"
    }
} else {
    Write-Host "Process: NOT STARTED"
}

if (Test-Path -LiteralPath $heartbeatPath) {
    Write-Host "Heartbeat:"
    Get-Content -LiteralPath $heartbeatPath
}

Write-Host "Recent stdout:"
Get-Content -LiteralPath $stdout -Tail 8 -ErrorAction SilentlyContinue
Write-Host "Recent stderr:"
Get-Content -LiteralPath $stderr -Tail 8 -ErrorAction SilentlyContinue

& nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total `
    --format=csv,noheader
