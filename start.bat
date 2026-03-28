@echo off
set PROJECT=C:\Users\Tommy\Documents\Coding Projects\forecast

echo Starting Docker infrastructure...
pushd "%PROJECT%"
docker compose up -d
popd

start "Consumer"  powershell -NoExit -ExecutionPolicy Bypass -Command "Set-Location '%PROJECT%'; .\.venv\Scripts\Activate.ps1; python streaming/consumer.py"
start "Producer"  powershell -NoExit -ExecutionPolicy Bypass -Command "Set-Location '%PROJECT%'; .\.venv\Scripts\Activate.ps1; python streaming/producer.py"
start "API"       powershell -NoExit -ExecutionPolicy Bypass -Command "Set-Location '%PROJECT%'; .\.venv\Scripts\Activate.ps1; uvicorn api.main:app --port 8000"
start "Dashboard" powershell -NoExit -ExecutionPolicy Bypass -Command "Set-Location '%PROJECT%'; .\.venv\Scripts\Activate.ps1; streamlit run dashboard/app.py"
