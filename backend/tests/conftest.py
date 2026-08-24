"""Shared pytest fixtures.

Using TestClient as a context manager ensures the FastAPI lifespan runs, so the
model is actually loaded before the tests hit the endpoints.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c
