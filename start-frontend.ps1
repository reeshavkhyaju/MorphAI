# Starts the Vite dev server on http://localhost:5173.
# /api and /predict are proxied to the Flask backend on port 5000.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontend = Join-Path $root 'frontend'

if (-not (Test-Path (Join-Path $frontend 'node_modules'))) {
    Write-Host 'Installing frontend dependencies...' -ForegroundColor Cyan
    Push-Location $frontend
    npm install
    Pop-Location
}

Push-Location $frontend
npm run dev
Pop-Location
