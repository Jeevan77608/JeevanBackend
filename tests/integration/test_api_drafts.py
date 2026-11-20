"""Integration tests for the drafts REST API endpoints.

Covers:
- GET  /api/v1/drafts  (list drafts)
- DELETE /api/v1/drafts  (clear drafts)

All tests run with AUTH_ENABLED=false so the mock-session path is used
and no real Azure AD tokens are required.
"""

from __future__ import annotations

import httpx
import pytest


# Uses shared app_client fixture from tests/integration/conftest.py


# =============================================================================
# GET /api/v1/drafts
# =============================================================================


async def test_list_drafts_empty(app_client: httpx.AsyncClient) -> None:
    """GET /api/v1/drafts returns an empty list when no drafts exist."""
    response = await app_client.get("/api/v1/drafts")

    assert response.status_code == 200
    assert response.json() == []


async def test_list_drafts_returns_list_format(app_client: httpx.AsyncClient) -> None:
    """GET /api/v1/drafts always returns a JSON array (even when empty)."""
    response = await app_client.get("/api/v1/drafts")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


async def test_list_drafts_content_type_is_json(app_client: httpx.AsyncClient) -> None:
    """GET /api/v1/drafts returns application/json content type."""
    response = await app_client.get("/api/v1/drafts")

    assert response.status_code == 200
    assert "application/json" in response.headers.get("content-type", "")


# =============================================================================
# DELETE /api/v1/drafts
# =============================================================================


async def test_clear_drafts_returns_message(app_client: httpx.AsyncClient) -> None:
    """DELETE /api/v1/drafts returns a success message."""
    response = await app_client.delete("/api/v1/drafts")

    assert response.status_code == 200
    body = response.json()
    assert "message" in body
    assert body["message"] == "All drafts cleared"


async def test_clear_drafts_idempotent(app_client: httpx.AsyncClient) -> None:
    """DELETE /api/v1/drafts is idempotent -- calling twice still succeeds."""
    first = await app_client.delete("/api/v1/drafts")
    second = await app_client.delete("/api/v1/drafts")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
