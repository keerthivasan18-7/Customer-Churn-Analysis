$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
	throw "Python virtual environment not found at $pythonExe. Create it first with: python -m venv .venv"
}

Set-Location $projectRoot
& $pythonExe -m streamlit run src\dashboard.py
