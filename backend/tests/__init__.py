"""
Shared test fixtures for PastCast backend tests.
"""

import os
import sys
import pytest

# Ensure backend is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["FLASK_ENV"] = "testing"


@pytest.fixture
def app():
    """Create a test Flask application with in-memory database."""
    from app import create_app
    from config import TestingConfig

    test_app = create_app(config=TestingConfig())
    yield test_app


@pytest.fixture
def client(app):
    """Create a test client for the Flask app."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """Create a CLI test runner."""
    return app.test_cli_runner()
