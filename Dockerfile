# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN pip install uv

# Copy necessary files for dependency installation
COPY pyproject.toml .
COPY README.md .

# --- ADDED: Copy source code so Hatchling can build the editable package layout ---
COPY src/ src/

# Install dependencies (and your project in editable mode)
RUN uv pip install --system -e .

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and other directories
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY data/ data/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/models/checkpoints

# Default command
CMD ["python", "scripts/generate_data.py"]
