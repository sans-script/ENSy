"""
conftest.py — Global Playwright test configuration
Project: Semantic Manifold Vector Visualizer
"""

import pytest


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def base_url() -> str:
    """Base URL for the Vite dev server (frontend)."""
    return "http://localhost:5173"


@pytest.fixture(scope="session")
def backend_url() -> str:
    """Base URL for the FastAPI backend."""
    return "http://127.0.0.1:3000"


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args: dict) -> dict:
    """
    Global browser context settings: fixed at 1920x1080 Full HD.
    """
    return {
        **browser_context_args,
        "viewport": {"width": 1920, "height": 1080},
        "locale": "en-US",
    }
