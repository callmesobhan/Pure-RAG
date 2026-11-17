# ============================================================================
# Pure RAG — Windows PowerShell Installer & Hardware Setup Wizard
# Usage:
#   Human interactive:   powershell -ExecutionPolicy Bypass -File .\install.ps1
#   AI Agent / Auto:     powershell -ExecutionPolicy Bypass -File .\install.ps1 -Yes
# ============================================================================
param(
    [switch]$Yes = $false,
    [switch]$Minimal = $false,
    [switch]$Serve = $false
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  Pure RAG — Local SOTA RAG Pipeline Installer (Windows)" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

# 1. Locate Python 3.9+
$pyCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pyCmd = "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pyCmd = "py"
} else {
    Write-Host "Error: Python 3.9+ was not found. Install Python from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

Write-Host "Found Python: $(& $pyCmd --version)" -ForegroundColor Green

# 2. Create .venv
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment in .\.venv ..."
    & $pyCmd -m venv --system-site-packages .venv
}

$venvPy = ".\.venv\Scripts\python.exe"
$venvPip = ".\.venv\Scripts\pip.exe"

# 3. Permission prompt for human users
if (-not $Yes) {
    $confirm = Read-Host "Grant permission to install Pure RAG dependencies into .\.venv? [Y/n]"
    if ($confirm -ne "" -and $confirm -notmatch "^[Yy]") {
        Write-Host "Installation cancelled."
        exit 0
    }
}

Write-Host "Installing Pure RAG core dependencies..." -ForegroundColor Cyan
& $venvPy -m pip install --upgrade pip wheel setuptools | Out-Null
& $venvPip install -e .

if (-not $Minimal) {
    $doSota = $true
    if (-not $Yes) {
        $ans = Read-Host "Install optional SOTA PDF & ONNX accelerators (pymupdf4llm, fastembed, flashrank)? [Y/n]"
        if ($ans -ne "" -and $ans -notmatch "^[Yy]") { $doSota = $false }
    }
    if ($doSota) {
        try {
            & $venvPip install ".[sota]"
        } catch {
            Write-Host "Continuing with core hybrid retriever..." -ForegroundColor Yellow
        }
    }
}

New-Item -ItemType Directory -Force -Path "pdfs" | Out-Null
New-Item -ItemType Directory -Force -Path "data" | Out-Null

if ($Yes) {
    & $venvPy -m purerag.cli setup --yes --ingest
} else {
    & $venvPy -m purerag.cli setup --ingest
}

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Green
Write-Host "  Pure RAG is installed and ready!" -ForegroundColor Green
Write-Host "  Drop your PDFs into: $PSScriptRoot\pdfs\"
Write-Host "  Start anytime with:  .\run.bat"
Write-Host "==================================================================" -ForegroundColor Green

if ($Serve) {
    & $venvPy -m purerag.cli serve
}
