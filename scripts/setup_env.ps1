$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $repoRoot '.venv'
$tempDir = Join-Path $venvPath 'temp'
$cacheDir = Join-Path $venvPath 'pip-cache'

New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null

$env:TEMP = $tempDir
$env:TMP = $tempDir
$env:PIP_CACHE_DIR = $cacheDir

if (Test-Path $venvPath) {
    try {
        py -3.10 -m venv $venvPath --clear
    } catch {
        Write-Warning "The existing virtual environment could not be cleared cleanly; continuing with it if possible."
    }
} else {
    py -3.10 -m venv $venvPath
}

$pythonExe = Join-Path $venvPath 'Scripts/python.exe'
& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install --prefer-binary --disable-pip-version-check --no-input -r (Join-Path $repoRoot 'requirements.txt')

@"
Virtual environment ready at $venvPath
Use:
  $pythonExe main.py
  $pythonExe -m streamlit run web_dashboard.py --server.port 8501 --server.address 0.0.0.0
"@ | Write-Host
