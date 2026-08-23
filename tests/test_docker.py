"""Tests for Docker deployment"""

from pathlib import Path

import pytest


def find_compose_file() -> Path | None:
    """Find docker-compose file with either .yaml or .yml extension"""
    for ext in [".yaml", ".yml"]:
        path = Path(f"docker-compose{ext}")
        if path.exists():
            return path
    return None


def test_dockerfile_exists():
    """Test that Dockerfile exists"""
    dockerfile = Path("Dockerfile")
    assert dockerfile.exists(), "Dockerfile not found"


def test_docker_compose_exists():
    """Test that docker-compose file exists (.yaml or .yml)"""
    compose_file = find_compose_file()
    assert compose_file is not None, "docker-compose.yaml or docker-compose.yml not found"


def test_dockerignore_exists():
    """Test that .dockerignore exists"""
    dockerignore = Path(".dockerignore")
    assert dockerignore.exists(), ".dockerignore not found"


def test_dockerignore_contents():
    """Test that .dockerignore has critical entries"""
    dockerignore = Path(".dockerignore")
    if not dockerignore.exists():
        pytest.skip(".dockerignore not found")

    content = dockerignore.read_text()

    assert ".venv" in content
    assert "__pycache__" in content
    assert ".git" in content
