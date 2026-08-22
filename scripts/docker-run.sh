#!/bin/bash
set -e

echo "=== PricePilot AI - One Command Deployment ==="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Build and start services
echo "Building Docker images..."
docker-compose build

echo "Starting services..."
docker-compose up -d

echo ""
echo "Waiting for services to be ready..."
sleep 10

echo ""
echo "=== Services Started ==="
echo "API:     http://localhost:8000"
echo "MLflow:  http://localhost:5000"
echo ""
echo "=== Quick Commands ==="
echo "Get recommendation:"
echo "  curl http://localhost:8000/recommendation"
echo ""
echo "Run pipeline once:"
echo "  docker-compose run --rm pipeline"
echo ""
echo "View logs:"
echo "  docker-compose logs -f api"
echo ""
echo "Stop services:"
echo "  docker-compose down"