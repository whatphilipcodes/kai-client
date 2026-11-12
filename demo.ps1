$scriptDir = $PSScriptRoot
if (-not $scriptDir) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}
Set-Location $scriptDir

Write-Host "Starting backend servers (This will take a while if they are not already running...)"

$remoteCmd = @'
cd ~/repos/kai-server
docker compose up -d --wait
echo "All containers are healthy and running!"
'@
ssh spark ($remoteCmd -replace "`r", "")

Write-Host "Starting Client... (Press Space to start recording and Space again to stop)"
uv run kai-client