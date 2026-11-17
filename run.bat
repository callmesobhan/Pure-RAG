@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo First-time run detected. Launching Pure RAG installer...
    powershell -ExecutionPolicy Bypass -File "%~dp0install.ps1"
)
".venv\Scripts\python.exe" -m purerag.cli serve %*
