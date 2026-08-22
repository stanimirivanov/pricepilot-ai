@echo off
echo === PricePilot AI - One Command Deployment ===
echo.

REM Check if Docker is running
docker info > nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running. Please start Docker Desktop.
    exit /b 1
)

echo Building Docker images...
docker-compose build

echo Starting services...
docker-compose up -d

echo.
echo Waiting for services to be ready...
timeout /t 10 /nobreak > nul

echo.
echo === Services Started ===
echo API:     http://localhost:8000
echo MLflow:  http://localhost:5000
echo.
echo === Quick Commands ===
echo Get recommendation:
echo   curl http://localhost:8000/recommendation
echo.
echo Run pipeline once:
echo   docker-compose run --rm pipeline
echo.
echo View logs:
echo   docker-compose logs -f api
echo.
echo Stop services:
echo   docker-compose down

pause