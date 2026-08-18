.PHONY: setup install dev test lint format typecheck generate-data pipeline check-mlflow clean docker-build docker-run

# Setup project
setup:
	uv venv --python 3.11
	uv sync

# Install dependencies
install:
	uv sync

# Check MLflow installation
check-mlflow:
	uv run python scripts/check_mlflow.py

# Run development server
dev:
	uv run uvicorn src.pricepilot.api.main:app --reload --port 8000

# Run tests
test:
	uv run pytest tests/ -v --cov=src/pricepilot --cov-report=term-missing

# Run linting and type checking
lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy src

# Format code and sort imports
format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

# Type checking
typecheck:
	uv run mypy src/

# Generate synthetic data
generate-data:
	uv run python scripts/generate_data.py

# Run pipeline
pipeline: check-mlflow
	uv run python scripts/run_phase1_pipeline.py

# Run mlflow
mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

# Clean up
clean:
	uv cache clean
	uv run python scripts/clean.py

# Docker
docker-build:
	docker build -t pricepilot-ai .

docker-run:
	docker run -p 8000:8000 pricepilot-ai