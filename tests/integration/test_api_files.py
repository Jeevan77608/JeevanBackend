"""Integration tests for the file upload REST API endpoint.

Covers POST /api/v1/files/upload with:
- Valid text and markdown files
- Unsupported MIME types (expect 400)
- Oversize Content-Length header (expect 400)

All tests run with AUTH_ENABLED=false so the mock-session path is used
and no real Azure AD tokens are required.
"""

from __future__ import annotations

import httpx
import pytest


# Uses shared app_client fixture from tests/integration/conftest.py


# =============================================================================
# POST /api/v1/files/upload -- valid uploads
# =============================================================================


async def test_upload_text_file(app_client: httpx.AsyncClient) -> None:
    """Upload a .txt file and verify the response contains expected fields."""
    files = {"file": ("test.txt", b"Hello world", "text/plain")}
    response = await app_client.post("/api/v1/files/upload", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["file_name"] == "test.txt"
    assert "Hello world" in body["content"]
    assert body["file_type"] == ".txt"


async def test_upload_markdown_file(app_client: httpx.AsyncClient) -> None:
    """Upload a .md file and verify it processes correctly."""
    md_content = b"# Heading\n\nSome **bold** text."
    files = {"file": ("readme.md", md_content, "text/markdown")}
    response = await app_client.post("/api/v1/files/upload", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["file_name"] == "readme.md"
    assert "Heading" in body["content"]
    assert body["file_type"] == ".md"


async def test_upload_text_file_message_field(app_client: httpx.AsyncClient) -> None:
    """Upload response includes a human-readable message field."""
    files = {"file": ("notes.txt", b"Some notes", "text/plain")}
    response = await app_client.post("/api/v1/files/upload", files=files)

    assert response.status_code == 200
    body = response.json()
    assert "message" in body
    assert "notes.txt" in body["message"]


# =============================================================================
# POST /api/v1/files/upload -- rejection cases
# =============================================================================


async def test_upload_unsupported_type_returns_400(
    app_client: httpx.AsyncClient,
) -> None:
    """Uploading a file with an unsupported MIME type returns 400."""
    files = {"file": ("archive.zip", b"PK\x03\x04fake", "application/zip")}
    response = await app_client.post("/api/v1/files/upload", files=files)

    assert response.status_code == 400
    body = response.json()
    assert "detail" in body
    assert "Unsupported file type" in body["detail"]


async def test_upload_oversize_returns_400(app_client: httpx.AsyncClient) -> None:
    """A Content-Length header exceeding 10 MB triggers a 400 before reading the body.

    We set the header to a value larger than the 10 MB limit but only send a
    small payload so the test runs instantly.
    """
    oversize_length = str(11 * 1024 * 1024)  # 11 MB
    files = {"file": ("big.txt", b"tiny", "text/plain")}
    response = await app_client.post(
        "/api/v1/files/upload",
        files=files,
        headers={"content-length": oversize_length},
    )

    assert response.status_code == 400
    body = response.json()
    assert "detail" in body
    assert "size" in body["detail"].lower() or "limit" in body["detail"].lower()
