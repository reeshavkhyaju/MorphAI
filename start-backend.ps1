# Starts the MorphAI Flask API on http://127.0.0.1:5000
# Serves frontend/dist as well when it has been built.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
    Write-Host "No virtualenv found at $python" -ForegroundColor Yellow
    Write-Host "Create one first:  python -m venv .venv   (then see README.md)" -ForegroundColor Yellow
    exit 1
}

& $python (Join-Path $root 'backend\app.py')
