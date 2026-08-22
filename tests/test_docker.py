"""Tests for Docker deployment"""

import os
import subprocess
from pathlib import Path

import pytest


def test_dockerfile_exists():
    """Test that Dockerfile exists"""
    dockerfile = Path("Dockerfile")
    assert dockerfile.exists(), "Dockerfile not found"


def test_docker_compose_exists():
    """Test that docker-compose.yml exists"""
    compose_file = Path("docker-compose.yml")
    assert compose_file.exists(), "docker-compose.yml not found"


def test_dockerignore_exists():
    """Test that .dockerignore exists"""
    dockerignore = Path(".dockerignore")
    assert dockerignore.exists(), ".dockerignore not found"


def test_dockerignore_contents():
    """Test that .dockerignore has critical entries"""
    dockerignore = Path(".dockerignore")
    content = dockerignore.read_text()

    # Should ignore venv and cache
    assert ".venv" in content
    assert "__pycache__" in content
    assert ".git" in content


def test_docker_compose_services():
    """Test that docker-compose has required services"""
    import yaml

    compose_file = Path("docker-compose.yml")
    with open(compose_file) as f:
        compose = yaml.safe_load(f)

    services = compose.get("services", {})

    assert "api" in services, "API service missing"
    assert "mlflow" in services, "MLflow service missing"
    assert "pipeline" in services, "Pipeline service missing"


@pytest.mark.skipif(
    os.environ.get("SKIP_DOCKER_TESTS", "true").lower() == "true",
    reason="Docker tests skipped by default. Set SKIP_DOCKER_TESTS=false to run.",
)
def test_docker_build():
    """Test Docker build (requires Docker running)"""
    result = subprocess.run(
        ["docker", "build", "-t", "pricepilot-ai:test", "."],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"Docker build failed: {result.stderr}"


@pytest.mark.skipif(
    os.environ.get("SKIP_DOCKER_TESTS", "true").lower() == "true",
    reason="Docker tests skipped by default. Set SKIP_DOCKER_TESTS=false to run.",
)
def test_docker_run():
    """Test Docker container runs"""
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "pricepilot-ai:test",
            "python",
            "-c",
            "import pricepilot; print('OK')",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0
    assert "OK" in result.stdout
