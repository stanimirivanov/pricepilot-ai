.PHONY: setup install dev test lint format generate-data pipeline clean docker-build docker-run

# Setup project
setup:
	uv venv --python 3.11
	uv sync

# Install dependencies
install:
	uv sync

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

# Generate synthetic data
generate-data:
	uv run python scripts/generate_data.py

# Run pipeline
pipeline:
	uv run python scripts/run_pipeline.py

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov

# Docker
docker-build:
	docker build -t pricepilot-ai .

docker-run:
	docker run -p 8000:8000 pricepilot-ai