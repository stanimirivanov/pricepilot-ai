"""Docker security validation tests"""

from pathlib import Path

import yaml


class TestDockerSecurity:
    """Security tests for Docker configuration"""

    def test_dockerfile_non_root_user(self):
        """Verify non-root user is created and used"""
        dockerfile = Path("Dockerfile").read_text()

        # Should create user
        assert "useradd" in dockerfile
        assert "appuser" in dockerfile

        # Should switch to non-root
        assert "USER appuser" in dockerfile

    def test_dockerfile_multistage(self):
        """Verify multi-stage build reduces attack surface"""
        dockerfile = Path("Dockerfile").read_text()

        assert "AS builder" in dockerfile
        assert "COPY --from=builder" in dockerfile

    def test_dockerfile_healthcheck(self):
        """Verify health check is configured"""
        dockerfile = Path("Dockerfile").read_text()

        assert "HEALTHCHECK" in dockerfile

    def test_compose_volumes(self):
        """Verify volume mounts for persistence"""
        compose = yaml.safe_load(Path("docker-compose.yml").read_text())

        for service_name in ["api", "pipeline"]:
            service = compose["services"].get(service_name, {})
            volumes = service.get("volumes", [])
            assert len(volumes) > 0, f"{service_name} should have volumes"

    def test_compose_no_privileged(self):
        """Verify no privileged containers"""
        compose = yaml.safe_load(Path("docker-compose.yml").read_text())

        for service_name, service in compose["services"].items():
            assert not service.get("privileged", False), f"{service_name} should not be privileged"

    def test_compose_healthchecks(self):
        """Verify health checks for services"""
        compose = yaml.safe_load(Path("docker-compose.yml").read_text())

        for service_name in ["api", "mlflow"]:
            service = compose["services"].get(service_name, {})
            assert "healthcheck" in service, f"{service_name} should have healthcheck"

    def test_dockerignore_no_secrets(self):
        """Verify .dockerignore excludes sensitive files"""
        dockerignore = Path(".dockerignore").read_text()

        sensitive_patterns = [".env", ".git", "*.pem", "*.key"]
        for pattern in sensitive_patterns:
            assert pattern in dockerignore, f"{pattern} should be in .dockerignore"


class TestVolumePersistence:
    """Tests for data persistence"""

    def test_data_volumes_defined(self):
        """Verify data volumes are defined"""
        compose = yaml.safe_load(Path("docker-compose.yml").read_text())

        volumes = compose.get("volumes", {})
        assert "data" in volumes
        assert "models" in volumes
        assert "mlflow" in volumes

    def test_api_data_volume(self):
        """Verify API mounts data volume"""
        compose = yaml.safe_load(Path("docker-compose.yml").read_text())

        api_service = compose["services"]["api"]
        volumes = api_service.get("volumes", [])

        # Check for data mount
        data_mounts = [v for v in volumes if "/app/data" in v]
        assert len(data_mounts) > 0
