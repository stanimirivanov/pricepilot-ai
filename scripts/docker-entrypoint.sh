#!/bin/bash
set -e

echo "=== PricePilot AI Container ==="
echo "Environment: ${ENVIRONMENT:-docker}"
echo "Starting service..."

# Check if we need to generate data
if [ ! -f "data/raw/car_wash_transactions.csv" ]; then
    echo "No data found. Generating synthetic data..."
    python scripts/generate_data.py
fi

# Run the specified command
exec "$@"