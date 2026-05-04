import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring real hardware (GPU, camera). "
        "Skipped in CI, run locally with: pytest -m integration"
    )
