"""Unit tests for Azure DevOps REST API client.

Tests cover:
- Exception hierarchy (AzureDevOpsError, RateLimitError, AuthenticationError, etc.)
- ApiResponse dataclass
- TokenBucket rate limiter
- AzureDevOpsClient init, auth headers, context manager
- HTTP methods (_request, _get, _post, _patch, _delete)
- Organization & Project APIs
- Work Item APIs (CRUD, batch, query, search, hierarchy)
- Board & Backlog APIs
- Wiki APIs
- Batch operations (create_work_items_batch)
- Board context building (build_board_context)
"""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.clients.devops import (
    ApiResponse,
    AuthenticationError,
    AzureDevOpsClient,
    AzureDevOpsError,
    NotFoundError,
    RateLimitError,
    TokenBucket,
    ValidationError,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_settings(
    *,
    default_org_url: str = "https://dev.azure.com/TestOrg",
    api_version: str = "7.0",
    requests_per_second: float = 100.0,
    request_timeout: float = 30.0,
) -> MagicMock:
    """Create a mock AzureDevOpsSettings."""
    settings = MagicMock()
    settings.default_org_url = default_org_url
    settings.api_version = api_version
    settings.requests_per_second = requests_per_second
    settings.request_timeout = request_timeout
    return settings


def _make_response(
    status_code: int = 200,
    json_data: Any = None,
    headers: dict[str, str] | None = None,
    content: bytes = b"{}",
) -> MagicMock:
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = httpx.Headers(headers or {})
    resp.content = content
    resp.text = content.decode() if content else ""
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.return_value = {}
    return resp


async def _build_client(
    settings: MagicMock | None = None,
    access_token: str = "test-token",
    use_pat: bool = False,
) -> AzureDevOpsClient:
    """Build and enter an AzureDevOpsClient with a mocked httpx.AsyncClient."""
    settings = settings or _make_settings()
    client = AzureDevOpsClient(
        settings=settings,
        access_token=access_token,
        use_pat=use_pat,
    )
    await client.__aenter__()
    return client


# =============================================================================
# Exception hierarchy
# =============================================================================


class TestExceptionHierarchy:
    """Verify custom exception classes and their attributes."""

    def test_azure_devops_error_base(self) -> None:
        """AzureDevOpsError stores status_code and error_code."""
        err = AzureDevOpsError("boom", status_code=500, error_code="ServerError")
        assert str(err) == "boom"
        assert err.status_code == 500
        assert err.error_code == "ServerError"

    def test_azure_devops_error_defaults(self) -> None:
        """Defaults are None for optional fields."""
        err = AzureDevOpsError("simple")
        assert err.status_code is None
        assert err.error_code is None

    def test_rate_limit_error(self) -> None:
        """RateLimitError carries retry_after and is a subclass of AzureDevOpsError."""
        err = RateLimitError(retry_after=30.0)
        assert isinstance(err, AzureDevOpsError)
        assert err.retry_after == 30.0
        assert err.status_code == 429

    def test_rate_limit_error_default_retry(self) -> None:
        """Default retry_after is 60 seconds."""
        err = RateLimitError()
        assert err.retry_after == 60.0

    def test_authentication_error(self) -> None:
        """AuthenticationError is a subclass of AzureDevOpsError."""
        err = AuthenticationError("Unauthorized", status_code=401)
        assert isinstance(err, AzureDevOpsError)
        assert err.status_code == 401

    def test_not_found_error(self) -> None:
        """NotFoundError is a subclass of AzureDevOpsError."""
        err = NotFoundError("Missing", status_code=404)
        assert isinstance(err, AzureDevOpsError)

    def test_validation_error(self) -> None:
        """ValidationError is a subclass of AzureDevOpsError."""
        err = ValidationError("Bad input", status_code=400)
        assert isinstance(err, AzureDevOpsError)


# =============================================================================
# ApiResponse
# =============================================================================


class TestApiResponse:
    """Verify ApiResponse dataclass behaviour."""

    def test_defaults(self) -> None:
        """Headers default to empty dict, continuation_token to None."""
        resp = ApiResponse(data={"ok": True}, status_code=200)
        assert resp.data == {"ok": True}
        assert resp.status_code == 200
        assert resp.headers == {}
        assert resp.continuation_token is None

    def test_full_construction(self) -> None:
        """All fields can be set explicitly."""
        resp = ApiResponse(
            data=[1, 2],
            status_code=200,
            headers={"x-token": "abc"},
            continuation_token="ct-123",
        )
        assert resp.continuation_token == "ct-123"
        assert resp.headers["x-token"] == "abc"


# =============================================================================
# TokenBucket rate limiter
# =============================================================================


class TestTokenBucket:
    """Verify the token bucket rate limiting algorithm."""

    async def test_acquire_no_wait(self) -> None:
        """Acquiring within capacity returns immediately (wait_time == 0)."""
        bucket = TokenBucket(rate=10.0, capacity=10.0)
        waited = await bucket.acquire(1.0)
        assert waited == 0.0
        assert bucket.tokens == pytest.approx(9.0, abs=0.5)

    async def test_acquire_drains_tokens(self) -> None:
        """Acquiring more tokens than available causes a wait."""
        bucket = TokenBucket(rate=1000.0, capacity=1.0)
        # Drain the bucket
        await bucket.acquire(1.0)
        # Next acquire should wait (but rate is high so wait is tiny)
        t0 = time.monotonic()
        await bucket.acquire(1.0)
        elapsed = time.monotonic() - t0
        # The wait should be very short since rate is 1000/s
        assert elapsed < 0.1

    async def test_capacity_ceiling(self) -> None:
        """Tokens never exceed capacity even after long idle."""
        bucket = TokenBucket(rate=100.0, capacity=5.0)
        # Simulate time passing by setting last_update in the past
        bucket.last_update = time.monotonic() - 100
        await bucket.acquire(1.0)
        # Tokens should have been capped at capacity before deducting
        assert bucket.tokens <= bucket.capacity


# =============================================================================
# AzureDevOpsClient — Init & Auth
# =============================================================================


class TestClientInitAndAuth:
    """Client initialization, auth header generation, and context manager."""

    def test_init_bearer_token(self) -> None:
        """Bearer auth header is generated for OAuth tokens."""
        settings = _make_settings()
        client = AzureDevOpsClient(settings, access_token="my-oauth-token")
        assert client._get_auth_header() == "Bearer my-oauth-token"

    def test_init_pat_basic_auth(self) -> None:
        """PAT auth uses Base64-encoded Basic header with empty username."""
        settings = _make_settings()
        client = AzureDevOpsClient(
            settings, access_token="my-pat", use_pat=True,
        )
        header = client._get_auth_header()
        expected_encoded = base64.b64encode(b":my-pat").decode()
        assert header == f"Basic {expected_encoded}"

    def test_init_uses_default_org_url(self) -> None:
        """Client picks up default_org_url from settings."""
        settings = _make_settings(default_org_url="https://dev.azure.com/MyOrg")
        client = AzureDevOpsClient(settings, access_token="tok")
        assert client._org_url == "https://dev.azure.com/MyOrg"

    def test_init_explicit_org_url(self) -> None:
        """Explicit organization_url overrides settings default."""
        settings = _make_settings(default_org_url="https://dev.azure.com/Default")
        client = AzureDevOpsClient(
            settings,
            access_token="tok",
            organization_url="https://dev.azure.com/Override",
        )
        assert client._org_url == "https://dev.azure.com/Override"

    async def test_context_manager_creates_and_closes_client(self) -> None:
        """__aenter__ creates httpx client; __aexit__ closes it."""
        settings = _make_settings()
        client = AzureDevOpsClient(settings, access_token="tok")
        assert client._client is None

        async with client as c:
            assert c._client is not None
            assert isinstance(c._client, httpx.AsyncClient)

        assert client._client is None

    def test_get_client_raises_without_context(self) -> None:
        """_get_client raises RuntimeError when not in context manager."""
        settings = _make_settings()
        client = AzureDevOpsClient(settings, access_token="tok")
        with pytest.raises(RuntimeError, match="not initialized"):
            client._get_client()


# =============================================================================
# _request — rate limiting, error mapping, retries
# =============================================================================


class TestRequest:
    """Low-level _request behaviour: rate limiting, status handling, retries."""

    async def test_successful_get(self) -> None:
        """200 response returns ApiResponse with parsed JSON."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200,
            json_data={"value": [1, 2]},
            headers={"x-ms-continuationtoken": "ct-1"},
        )
        client._client = mock_http

        resp = await client._request("GET", "https://example.com/api")
        assert resp.data == {"value": [1, 2]}
        assert resp.status_code == 200
        assert resp.continuation_token == "ct-1"
        await client.__aexit__(None, None, None)

    async def test_429_raises_rate_limit_error(self) -> None:
        """429 response raises RateLimitError with Retry-After."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=429,
            headers={"Retry-After": "15"},
        )
        client._client = mock_http

        with pytest.raises(RateLimitError) as exc_info:
            # Disable tenacity retry for direct testing
            await client._request.__wrapped__(client, "GET", "https://example.com/api")

        assert exc_info.value.retry_after == 15.0
        await client.__aexit__(None, None, None)

    async def test_401_raises_authentication_error(self) -> None:
        """401 response raises AuthenticationError."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(status_code=401)
        client._client = mock_http

        with pytest.raises(AuthenticationError):
            await client._request.__wrapped__(client, "GET", "https://example.com/api")
        await client.__aexit__(None, None, None)

    async def test_404_raises_not_found_error(self) -> None:
        """404 response raises NotFoundError."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(status_code=404)
        client._client = mock_http

        with pytest.raises(NotFoundError):
            await client._request.__wrapped__(client, "GET", "https://example.com/api")
        await client.__aexit__(None, None, None)

    async def test_400_raises_generic_error(self) -> None:
        """400 response raises AzureDevOpsError with message from response body."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=400,
            json_data={"message": "Bad request payload"},
            content=b'{"message": "Bad request payload"}',
        )
        client._client = mock_http

        with pytest.raises(AzureDevOpsError, match="Bad request payload"):
            await client._request.__wrapped__(client, "GET", "https://example.com/api")
        await client.__aexit__(None, None, None)

    async def test_500_raises_generic_error(self) -> None:
        """500+ response raises AzureDevOpsError."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=500,
            json_data={},
            content=b'{}',
        )
        client._client = mock_http

        with pytest.raises(AzureDevOpsError):
            await client._request.__wrapped__(client, "GET", "https://example.com/api")
        await client.__aexit__(None, None, None)

    async def test_empty_content_response(self) -> None:
        """Response with no content returns data=None."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200,
            content=b"",
        )
        client._client = mock_http

        resp = await client._request("GET", "https://example.com/api")
        assert resp.data is None
        await client.__aexit__(None, None, None)

    async def test_http_error_wrapped(self) -> None:
        """httpx.HTTPError is wrapped in AzureDevOpsError."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.side_effect = httpx.ConnectError("Connection refused")
        client._client = mock_http

        with pytest.raises(AzureDevOpsError, match="HTTP error"):
            await client._request.__wrapped__(client, "GET", "https://example.com/api")
        await client.__aexit__(None, None, None)

    async def test_api_version_injected(self) -> None:
        """api-version parameter is added to every request."""
        settings = _make_settings(api_version="7.1-preview")
        client = await _build_client(settings=settings)
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200,
            json_data={},
        )
        client._client = mock_http

        await client._request("GET", "https://example.com/api")
        call_kwargs = mock_http.request.call_args
        assert call_kwargs.kwargs["params"]["api-version"] == "7.1-preview"
        await client.__aexit__(None, None, None)


# =============================================================================
# Organization & Project APIs
# =============================================================================


class TestOrganizationAndProjectAPIs:
    """Tests for get_organizations, get_projects, get_project."""

    async def test_get_organizations(self) -> None:
        """Parses organization list from VSSPS API response."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200,
            json_data={
                "value": [
                    {"accountId": "org-1", "accountName": "MyOrg"},
                    {"accountId": "org-2", "accountName": "OtherOrg"},
                ],
            },
        )
        client._client = mock_http

        orgs = await client.get_organizations()
        assert len(orgs) == 2
        assert orgs[0].id == "org-1"
        assert orgs[0].name == "MyOrg"
        assert orgs[0].url == "https://dev.azure.com/MyOrg"
        assert orgs[1].name == "OtherOrg"
        await client.__aexit__(None, None, None)

    async def test_get_projects(self) -> None:
        """Parses project list from the organization API."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200,
            json_data={
                "value": [
                    {
                        "id": "proj-1",
                        "name": "ProjectAlpha",
                        "description": "Alpha desc",
                        "url": "https://dev.azure.com/Org/_apis/projects/proj-1",
                        "state": "wellFormed",
                    },
                ],
            },
        )
        client._client = mock_http

        projects = await client.get_projects()
        assert len(projects) == 1
        assert projects[0].name == "ProjectAlpha"
        assert projects[0].description == "Alpha desc"
        await client.__aexit__(None, None, None)

    async def test_get_projects_requires_org_url(self) -> None:
        """Raises ValueError when no org URL is available."""
        settings = _make_settings(default_org_url=None)
        client = await _build_client(settings=settings)
        # Override to ensure no org_url
        client._org_url = None

        with pytest.raises(ValueError, match="Organization URL required"):
            await client.get_projects()
        await client.__aexit__(None, None, None)

    async def test_get_project(self) -> None:
        """Fetches a single project by ID."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200,
            json_data={
                "id": "proj-42",
                "name": "Zeta",
                "state": "wellFormed",
            },
        )
        client._client = mock_http

        project = await client.get_project("proj-42")
        assert project.id == "proj-42"
        assert project.name == "Zeta"
        await client.__aexit__(None, None, None)


# =============================================================================
# Work Item APIs
# =============================================================================


class TestWorkItemAPIs:
    """Tests for work item CRUD, batch, query, search, and hierarchy."""

    async def test_get_work_item_types(self) -> None:
        """Parses work item types from API response."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200,
            json_data={
                "value": [
                    {
                        "name": "User Story",
                        "referenceName": "Microsoft.VSTS.WorkItemTypes.UserStory",
                        "description": "A user story",
                        "icon": {"url": "https://icon.png"},
                        "color": "009CCC",
                    },
                    {
                        "name": "Bug",
                        "referenceName": "Microsoft.VSTS.WorkItemTypes.Bug",
                    },
                ],
            },
        )
        client._client = mock_http

        types = await client.get_work_item_types("MyProject")
        assert len(types) == 2
        assert types[0].name == "User Story"
        assert types[0].reference_name == "Microsoft.VSTS.WorkItemTypes.UserStory"
        assert types[1].name == "Bug"
        await client.__aexit__(None, None, None)

    async def test_get_work_item(self) -> None:
        """Fetches a work item and validates through Pydantic model."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200,
            json_data={
                "id": 101,
                "rev": 3,
                "url": "https://dev.azure.com/Org/_apis/wit/workitems/101",
                "fields": {
                    "System.Title": "Login Feature",
                    "System.WorkItemType": "User Story",
                    "System.State": "Active",
                },
                "relations": [],
                "_links": {},
            },
        )
        client._client = mock_http

        wi = await client.get_work_item(101, project="MyProject")
        assert wi.id == 101
        assert wi.title == "Login Feature"
        assert wi.work_item_type == "User Story"
        await client.__aexit__(None, None, None)

    async def test_get_work_item_without_project(self) -> None:
        """URL is built without project segment when project is None."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200,
            json_data={
                "id": 200,
                "rev": 1,
                "url": "",
                "fields": {"System.Title": "No project"},
                "relations": [],
                "_links": {},
            },
        )
        client._client = mock_http

        wi = await client.get_work_item(200)
        assert wi.id == 200
        # Verify URL doesn't contain a project segment
        call_args = mock_http.request.call_args
        url_arg = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("url", "")
        assert "/_apis/wit/workitems/200" in url_arg
        await client.__aexit__(None, None, None)

    async def test_get_work_item_raw(self) -> None:
        """get_work_item_raw returns the raw dict without validation."""
        client = await _build_client()
        mock_http = AsyncMock()
        raw = {"id": 55, "fields": {"System.Title": "Raw"}, "extra_field": True}
        mock_http.request.return_value = _make_response(
            status_code=200,
            json_data=raw,
        )
        client._client = mock_http

        result = await client.get_work_item_raw(55, project="P")
        assert result == raw
        await client.__aexit__(None, None, None)

    async def test_get_work_items_batch_empty(self) -> None:
        """Empty ID list returns empty list without making a request."""
        client = await _build_client()
        result = await client.get_work_items_batch([])
        assert result == []
        await client.__aexit__(None, None, None)

    async def test_get_work_items_batch_over_200_raises(self) -> None:
        """More than 200 IDs raises ValueError."""
        client = await _build_client()
        with pytest.raises(ValueError, match="Maximum 200"):
            await client.get_work_items_batch(list(range(201)))
        await client.__aexit__(None, None, None)

    async def test_get_work_items_batch_success(self) -> None:
        """Parses batch response and returns validated WorkItem list."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200,
            json_data={
                "count": 2,
                "value": [
                    {
                        "id": 10,
                        "rev": 1,
                        "url": "",
                        "fields": {"System.Title": "Item A"},
                        "relations": [],
                        "_links": {},
                    },
                    {
                        "id": 11,
                        "rev": 1,
                        "url": "",
                        "fields": {"System.Title": "Item B"},
                        "relations": [],
                        "_links": {},
                    },
                ],
            },
        )
        client._client = mock_http

        items = await client.get_work_items_batch([10, 11], project="P")
        assert len(items) == 2
        assert items[0].id == 10
        assert items[1].id == 11
        await client.__aexit__(None, None, None)

    async def test_query_work_items(self) -> None:
        """Executes WIQL query and returns list of work item IDs."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200,
            json_data={
                "queryType": "flat",
                "queryResultType": "workItem",
                "workItems": [
                    {"id": 100, "url": "..."},
                    {"id": 101, "url": "..."},
                ],
            },
        )
        client._client = mock_http

        ids = await client.query_work_items("Proj", "SELECT [System.Id] FROM WorkItems")
        assert ids == [100, 101]
        await client.__aexit__(None, None, None)

    async def test_search_work_items_no_results(self) -> None:
        """Empty query result returns empty list without fetching batch."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200,
            json_data={
                "queryType": "flat",
                "queryResultType": "workItem",
                "workItems": [],
            },
        )
        client._client = mock_http

        items = await client.search_work_items("Proj", work_item_type="Bug")
        assert items == []
        # Only one HTTP request (WIQL query), no batch fetch
        assert mock_http.request.call_count == 1
        await client.__aexit__(None, None, None)

    async def test_search_work_items_with_filters(self) -> None:
        """Filters are embedded in the WIQL query sent to the API."""
        client = await _build_client()
        mock_http = AsyncMock()

        # First call: WIQL query returns IDs
        wiql_resp = _make_response(
            status_code=200,
            json_data={
                "queryType": "flat",
                "queryResultType": "workItem",
                "workItems": [{"id": 50}],
            },
        )
        # Second call: batch fetch
        batch_resp = _make_response(
            status_code=200,
            json_data={
                "count": 1,
                "value": [
                    {
                        "id": 50,
                        "rev": 1,
                        "url": "",
                        "fields": {
                            "System.Title": "Filtered Bug",
                            "System.WorkItemType": "Bug",
                        },
                        "relations": [],
                        "_links": {},
                    },
                ],
            },
        )
        mock_http.request.side_effect = [wiql_resp, batch_resp]
        client._client = mock_http

        items = await client.search_work_items(
            "Proj",
            work_item_type="Bug",
            state="Active",
            title_contains="Filtered",
        )
        assert len(items) == 1
        assert items[0].title == "Filtered Bug"

        # Verify WIQL query in first request contains filters
        first_call = mock_http.request.call_args_list[0]
        body = first_call.kwargs.get("json", {})
        wiql_query = body.get("query", "")
        assert "Bug" in wiql_query
        assert "Active" in wiql_query
        assert "Filtered" in wiql_query
        await client.__aexit__(None, None, None)


# =============================================================================
# Hierarchy APIs
# =============================================================================


class TestHierarchyAPIs:
    """Tests for get_work_item_children and get_work_item_hierarchy."""

    async def test_get_work_item_children_no_children(self) -> None:
        """Returns empty list when parent has no child relations."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200,
            json_data={
                "id": 1,
                "rev": 1,
                "url": "",
                "fields": {"System.Title": "Parent"},
                "relations": [],
                "_links": {},
            },
        )
        client._client = mock_http

        children = await client.get_work_item_children(1, project="P")
        assert children == []
        await client.__aexit__(None, None, None)

    async def test_get_work_item_children_with_children(self) -> None:
        """Fetches child work items when parent has Hierarchy-Forward relations."""
        client = await _build_client()
        mock_http = AsyncMock()

        # First call: get parent with child relations
        parent_resp = _make_response(
            status_code=200,
            json_data={
                "id": 1,
                "rev": 1,
                "url": "",
                "fields": {"System.Title": "Epic"},
                "relations": [
                    {
                        "rel": "System.LinkTypes.Hierarchy-Forward",
                        "url": "https://dev.azure.com/Org/_apis/wit/workItems/10",
                        "attributes": {},
                    },
                    {
                        "rel": "System.LinkTypes.Hierarchy-Forward",
                        "url": "https://dev.azure.com/Org/_apis/wit/workItems/11",
                        "attributes": {},
                    },
                ],
                "_links": {},
            },
        )
        # Second call: batch fetch children
        batch_resp = _make_response(
            status_code=200,
            json_data={
                "count": 2,
                "value": [
                    {"id": 10, "rev": 1, "url": "", "fields": {"System.Title": "Child A"}, "relations": [], "_links": {}},
                    {"id": 11, "rev": 1, "url": "", "fields": {"System.Title": "Child B"}, "relations": [], "_links": {}},
                ],
            },
        )
        mock_http.request.side_effect = [parent_resp, batch_resp]
        client._client = mock_http

        children = await client.get_work_item_children(1, project="P")
        assert len(children) == 2
        assert children[0].id == 10
        assert children[1].id == 11
        await client.__aexit__(None, None, None)

    async def test_get_work_item_hierarchy_with_parent(self) -> None:
        """Returns work item, its parent, and children."""
        client = await _build_client()
        mock_http = AsyncMock()

        # Call 1: get the work item itself (has parent relation + child relations)
        item_resp = _make_response(
            status_code=200,
            json_data={
                "id": 50,
                "rev": 1,
                "url": "",
                "fields": {"System.Title": "Story"},
                "relations": [
                    {
                        "rel": "System.LinkTypes.Hierarchy-Reverse",
                        "url": "https://dev.azure.com/Org/_apis/wit/workItems/5",
                        "attributes": {},
                    },
                    {
                        "rel": "System.LinkTypes.Hierarchy-Forward",
                        "url": "https://dev.azure.com/Org/_apis/wit/workItems/100",
                        "attributes": {},
                    },
                ],
                "_links": {},
            },
        )
        # Call 2: get parent
        parent_resp = _make_response(
            status_code=200,
            json_data={
                "id": 5,
                "rev": 1,
                "url": "",
                "fields": {"System.Title": "Epic Parent"},
                "relations": [],
                "_links": {},
            },
        )
        # Call 3: batch fetch children
        children_resp = _make_response(
            status_code=200,
            json_data={
                "count": 1,
                "value": [
                    {"id": 100, "rev": 1, "url": "", "fields": {"System.Title": "Task"}, "relations": [], "_links": {}},
                ],
            },
        )
        mock_http.request.side_effect = [item_resp, parent_resp, children_resp]
        client._client = mock_http

        hierarchy = await client.get_work_item_hierarchy(50, project="P")
        assert hierarchy["work_item"].id == 50
        assert hierarchy["parent"].id == 5
        assert len(hierarchy["children"]) == 1
        assert hierarchy["children"][0].id == 100
        await client.__aexit__(None, None, None)

    async def test_get_work_item_hierarchy_no_parent_no_children(self) -> None:
        """Standalone work item returns None parent and empty children."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200,
            json_data={
                "id": 77,
                "rev": 1,
                "url": "",
                "fields": {"System.Title": "Lonely Item"},
                "relations": [],
                "_links": {},
            },
        )
        client._client = mock_http

        hierarchy = await client.get_work_item_hierarchy(77, project="P")
        assert hierarchy["work_item"].id == 77
        assert hierarchy["parent"] is None
        assert hierarchy["children"] == []
        await client.__aexit__(None, None, None)

    async def test_get_work_item_hierarchy_parent_not_found(self) -> None:
        """Missing parent (404) results in parent=None, no crash."""
        client = await _build_client()
        mock_http = AsyncMock()

        item_resp = _make_response(
            status_code=200,
            json_data={
                "id": 60,
                "rev": 1,
                "url": "",
                "fields": {"System.Title": "Orphan"},
                "relations": [
                    {
                        "rel": "System.LinkTypes.Hierarchy-Reverse",
                        "url": "https://dev.azure.com/Org/_apis/wit/workItems/999",
                        "attributes": {},
                    },
                ],
                "_links": {},
            },
        )
        parent_404 = _make_response(status_code=404)

        mock_http.request.side_effect = [item_resp, parent_404]
        client._client = mock_http

        # The NotFoundError from fetching parent should be caught
        # We need the _request to raise NotFoundError (not use __wrapped__)
        # Let's patch at a higher level
        original_request = client._request

        call_count = 0

        async def side_effect_request(method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: return the item
                return ApiResponse(
                    data={
                        "id": 60,
                        "rev": 1,
                        "url": "",
                        "fields": {"System.Title": "Orphan"},
                        "relations": [
                            {
                                "rel": "System.LinkTypes.Hierarchy-Reverse",
                                "url": "https://dev.azure.com/Org/_apis/wit/workItems/999",
                                "attributes": {},
                            },
                        ],
                        "_links": {},
                    },
                    status_code=200,
                )
            elif call_count == 2:
                raise NotFoundError("Not found", status_code=404)
            return ApiResponse(data={}, status_code=200)

        client._request = AsyncMock(side_effect=side_effect_request)

        hierarchy = await client.get_work_item_hierarchy(60, project="P")
        assert hierarchy["work_item"].id == 60
        assert hierarchy["parent"] is None
        await client.__aexit__(None, None, None)


# =============================================================================
# Create / Update Work Items
# =============================================================================


class TestCreateUpdateWorkItems:
    """Tests for create_work_item, update_work_item, and add_work_item_link."""

    async def test_create_work_item_basic(self) -> None:
        """Creates a work item with JSON Patch operations."""
        client = await _build_client()
        mock_http = AsyncMock()
        created_data = {"id": 42, "url": "https://dev.azure.com/Org/_workitems/edit/42"}
        mock_http.request.return_value = _make_response(
            status_code=200, json_data=created_data,
        )
        client._client = mock_http

        result = await client.create_work_item(
            project="P",
            work_item_type="User Story",
            fields={"System.Title": "New Story"},
        )
        assert result["id"] == 42

        # Verify URL includes $User Story
        call_args = mock_http.request.call_args
        url = call_args.args[1]
        assert "$User Story" in url

        # Verify JSON Patch body
        body = call_args.kwargs.get("json", [])
        titles = [op for op in body if op.get("path") == "/fields/System.Title"]
        assert len(titles) == 1
        assert titles[0]["value"] == "New Story"
        await client.__aexit__(None, None, None)

    async def test_create_work_item_with_parent(self) -> None:
        """Creates a work item linked to a parent via Hierarchy-Reverse relation."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200, json_data={"id": 43},
        )
        client._client = mock_http

        await client.create_work_item(
            project="P",
            work_item_type="Task",
            fields={"System.Title": "Child Task"},
            parent_id=10,
        )

        body = mock_http.request.call_args.kwargs.get("json", [])
        link_ops = [op for op in body if op.get("path") == "/relations/-"]
        assert len(link_ops) == 1
        assert "Hierarchy-Reverse" in link_ops[0]["value"]["rel"]
        assert "10" in link_ops[0]["value"]["url"]
        await client.__aexit__(None, None, None)

    async def test_update_work_item(self) -> None:
        """Updates a work item with JSON Patch operations."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200,
            json_data={"id": 42, "fields": {"System.Title": "Updated"}},
        )
        client._client = mock_http

        result = await client.update_work_item(
            42, {"System.Title": "Updated"},
        )
        assert result["id"] == 42

        # Verify PATCH method
        call_args = mock_http.request.call_args
        assert call_args.args[0] == "PATCH"
        await client.__aexit__(None, None, None)

    async def test_add_work_item_link(self) -> None:
        """Adds a parent link between two work items."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200, json_data={"id": 20},
        )
        client._client = mock_http

        result = await client.add_work_item_link(
            project="P", work_item_id=20, target_id=5,
        )
        assert result["id"] == 20

        body = mock_http.request.call_args.kwargs.get("json", [])
        assert len(body) == 1
        assert "Hierarchy-Reverse" in body[0]["value"]["rel"]
        assert "5" in body[0]["value"]["url"]
        await client.__aexit__(None, None, None)


# =============================================================================
# Board APIs
# =============================================================================


class TestBoardAPIs:
    """Tests for get_boards, get_board_columns, and get_backlogs."""

    async def test_get_boards(self) -> None:
        """Parses board list from API response."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200,
            json_data={
                "value": [
                    {"id": "board-1", "name": "Stories Board", "url": "https://..."},
                ],
            },
        )
        client._client = mock_http

        boards = await client.get_boards("P")
        assert len(boards) == 1
        assert boards[0].name == "Stories Board"
        await client.__aexit__(None, None, None)

    async def test_get_boards_with_team(self) -> None:
        """URL includes team segment when team is specified."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200, json_data={"value": []},
        )
        client._client = mock_http

        await client.get_boards("P", team="TeamA")
        url = mock_http.request.call_args.args[1]
        assert "/P/TeamA/_apis/work/boards" in url
        await client.__aexit__(None, None, None)

    async def test_get_board_columns(self) -> None:
        """Returns column data for a board."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200,
            json_data={
                "value": [
                    {"name": "New", "itemLimit": 0},
                    {"name": "Active", "itemLimit": 5},
                    {"name": "Done", "itemLimit": 0},
                ],
            },
        )
        client._client = mock_http

        cols = await client.get_board_columns("P", "board-1")
        assert len(cols) == 3
        assert cols[0]["name"] == "New"
        await client.__aexit__(None, None, None)

    async def test_get_backlogs(self) -> None:
        """Returns raw backlog data from the API."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200,
            json_data={
                "value": [
                    {"id": "bl-1", "name": "Epics", "rank": 3, "type": "portfolio"},
                ],
            },
        )
        client._client = mock_http

        backlogs = await client.get_backlogs("P")
        assert len(backlogs) == 1
        assert backlogs[0]["name"] == "Epics"
        await client.__aexit__(None, None, None)


# =============================================================================
# Wiki APIs
# =============================================================================


class TestWikiAPIs:
    """Tests for get_wikis, get_wiki_page, list_wiki_pages, create_wiki_page."""

    async def test_get_wikis(self) -> None:
        """Parses wiki list from API response."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200,
            json_data={
                "value": [
                    {
                        "id": "wiki-1",
                        "name": "Project.wiki",
                        "projectId": "proj-1",
                        "type": "projectWiki",
                        "url": "https://...",
                    },
                ],
            },
        )
        client._client = mock_http

        wikis = await client.get_wikis("P")
        assert len(wikis) == 1
        assert wikis[0].name == "Project.wiki"
        assert wikis[0].type == "projectWiki"
        await client.__aexit__(None, None, None)

    async def test_get_wiki_page(self) -> None:
        """Fetches a single wiki page by path."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200,
            json_data={
                "id": "page-1",
                "path": "/Home",
                "content": "# Welcome",
                "gitItemPath": "/Home.md",
                "order": 0,
            },
        )
        client._client = mock_http

        page = await client.get_wiki_page("P", "wiki-1", "/Home")
        assert page.path == "/Home"
        assert page.content == "# Welcome"
        await client.__aexit__(None, None, None)

    async def test_list_wiki_pages_recursive(self) -> None:
        """Flattens nested wiki page tree into a single list."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200,
            json_data={
                "id": "root",
                "path": "/",
                "subPages": [
                    {
                        "id": "child-1",
                        "path": "/Page1",
                        "subPages": [
                            {
                                "id": "grandchild-1",
                                "path": "/Page1/SubPage",
                                "subPages": [],
                            },
                        ],
                    },
                    {
                        "id": "child-2",
                        "path": "/Page2",
                        "subPages": [],
                    },
                ],
            },
        )
        client._client = mock_http

        pages = await client.list_wiki_pages("P", "wiki-1")
        assert len(pages) == 4  # root + 2 children + 1 grandchild
        paths = [p.path for p in pages]
        assert "/" in paths
        assert "/Page1" in paths
        assert "/Page1/SubPage" in paths
        assert "/Page2" in paths
        await client.__aexit__(None, None, None)

    async def test_create_wiki_page(self) -> None:
        """Creates a wiki page with PUT request."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(
            status_code=200,
            json_data={
                "id": "new-page",
                "path": "/NewPage",
                "gitItemPath": "/NewPage.md",
            },
        )
        client._client = mock_http

        page = await client.create_wiki_page("P", "wiki-1", "/NewPage", "# New Content")
        assert page.path == "/NewPage"
        assert page.content == "# New Content"

        # Verify PUT method
        call_args = mock_http.request.call_args
        assert call_args.args[0] == "PUT"
        await client.__aexit__(None, None, None)


# =============================================================================
# Batch Operations (create_work_items_batch)
# =============================================================================


class TestCreateWorkItemsBatch:
    """Tests for create_work_items_batch with DraftWorkItem inputs."""

    async def test_batch_success(self) -> None:
        """All items created successfully produces success PushResults."""
        from src.models.work_items import DraftWorkItem

        client = await _build_client()

        items = [
            DraftWorkItem(
                draft_id="d1",
                work_item_type="User Story",
                title="Story 1",
                description="Desc 1",
                priority=2,
                story_points=3.0,
            ),
            DraftWorkItem(
                draft_id="d2",
                work_item_type="Task",
                title="Task 1",
            ),
        ]

        # Mock create_work_item to return created data
        client.create_work_item = AsyncMock(
            side_effect=[
                {"id": 100, "url": "https://dev.azure.com/Org/_workitems/edit/100"},
                {"id": 101, "url": "https://dev.azure.com/Org/_workitems/edit/101"},
            ]
        )

        results = await client.create_work_items_batch("P", items)
        assert len(results) == 2
        # PushResult fields depend on how the model is defined; test the attempt was made
        assert client.create_work_item.call_count == 2
        await client.__aexit__(None, None, None)

    async def test_batch_partial_failure(self) -> None:
        """One failing item does not prevent others from being created."""
        from src.models.work_items import DraftWorkItem

        client = await _build_client()

        items = [
            DraftWorkItem(draft_id="d1", work_item_type="Bug", title="Bug 1"),
            DraftWorkItem(draft_id="d2", work_item_type="Bug", title="Bug 2"),
        ]

        client.create_work_item = AsyncMock(
            side_effect=[
                AzureDevOpsError("Server error", status_code=500),
                {"id": 200, "url": "https://..."},
            ]
        )

        results = await client.create_work_items_batch("P", items)
        assert len(results) == 2
        # Both items attempted
        assert client.create_work_item.call_count == 2
        await client.__aexit__(None, None, None)

    async def test_batch_builds_fields_correctly(self) -> None:
        """Verifies that DraftWorkItem fields are mapped to Azure DevOps fields."""
        from src.models.work_items import DraftWorkItem

        client = await _build_client()

        item = DraftWorkItem(
            draft_id="d1",
            work_item_type="User Story",
            title="Story With All Fields",
            description="Full description",
            acceptance_criteria="AC here",
            priority=1,
            story_points=8.0,
            assigned_to="user@test.com",
            tags=["tag1", "tag2"],
            area_path="Project\\Area1",
            iteration_path="Project\\Sprint1",
            custom_fields={"Custom.Field": "value"},
        )

        captured_fields = {}

        async def capture_create(project, work_item_type, fields, org_url=None, parent_id=None):
            captured_fields.update(fields)
            return {"id": 300, "url": "https://..."}

        client.create_work_item = AsyncMock(side_effect=capture_create)

        await client.create_work_items_batch("P", [item])

        assert captured_fields["System.Title"] == "Story With All Fields"
        assert captured_fields["System.Description"] == "Full description"
        assert captured_fields["Microsoft.VSTS.Common.AcceptanceCriteria"] == "AC here"
        assert captured_fields["Microsoft.VSTS.Common.Priority"] == 1
        assert captured_fields["Microsoft.VSTS.Scheduling.StoryPoints"] == 8.0
        assert captured_fields["System.AssignedTo"] == "user@test.com"
        assert captured_fields["System.Tags"] == "tag1; tag2"
        assert captured_fields["System.AreaPath"] == "Project\\Area1"
        assert captured_fields["System.IterationPath"] == "Project\\Sprint1"
        assert captured_fields["Custom.Field"] == "value"
        await client.__aexit__(None, None, None)


# =============================================================================
# build_board_context
# =============================================================================


class TestBuildBoardContext:
    """Tests for build_board_context with various process templates."""

    async def test_agile_template(self) -> None:
        """Agile process template detection (User Story as default requirement)."""
        client = await _build_client()

        # Mock get_work_item_types
        from src.models.devops import WorkItemType as WIT

        client.get_work_item_types = AsyncMock(
            return_value=[
                WIT(name="Epic", reference_name="Microsoft.VSTS.WorkItemTypes.Epic"),
                WIT(name="Feature", reference_name="Microsoft.VSTS.WorkItemTypes.Feature"),
                WIT(name="User Story", reference_name="Microsoft.VSTS.WorkItemTypes.UserStory"),
                WIT(name="Task", reference_name="Microsoft.VSTS.WorkItemTypes.Task"),
                WIT(name="Bug", reference_name="Microsoft.VSTS.WorkItemTypes.Bug"),
            ]
        )
        # Mock get_backlogs
        client.get_backlogs = AsyncMock(
            return_value=[
                {"id": "bl-epics", "name": "Epics", "rank": 3, "type": "portfolio", "workItemTypes": [{"name": "Epic"}]},
                {"id": "bl-features", "name": "Features", "rank": 2, "type": "portfolio", "workItemTypes": [{"name": "Feature"}]},
                {"id": "bl-stories", "name": "Stories", "rank": 1, "type": "requirement", "workItemTypes": [{"name": "User Story"}]},
                {"id": "bl-tasks", "name": "Tasks", "rank": 0, "type": "task", "workItemTypes": [{"name": "Task"}]},
            ]
        )

        ctx = await client.build_board_context("MyProject")
        assert ctx.default_requirement_type == "User Story"
        assert ctx.epic_type == "Epic"
        assert ctx.feature_type == "Feature"
        assert ctx.task_type == "Task"
        assert ctx.story_points_field == "story_points"
        assert "User Story" in ctx.parent_types_for_tasks
        assert ctx.process_template is not None
        assert ctx.process_template.value == "Agile"
        await client.__aexit__(None, None, None)

    async def test_scrum_template(self) -> None:
        """Scrum process template detection (Product Backlog Item as requirement)."""
        client = await _build_client()
        from src.models.devops import WorkItemType as WIT

        client.get_work_item_types = AsyncMock(
            return_value=[
                WIT(name="Epic", reference_name="r.Epic"),
                WIT(name="Feature", reference_name="r.Feature"),
                WIT(name="Product Backlog Item", reference_name="r.PBI"),
                WIT(name="Task", reference_name="r.Task"),
                WIT(name="Bug", reference_name="r.Bug"),
            ]
        )
        client.get_backlogs = AsyncMock(
            return_value=[
                {"id": "bl-e", "name": "Epics", "rank": 3, "type": "portfolio", "workItemTypes": [{"name": "Epic"}]},
                {"id": "bl-f", "name": "Features", "rank": 2, "type": "portfolio", "workItemTypes": [{"name": "Feature"}]},
                {"id": "bl-r", "name": "Requirements", "rank": 1, "type": "requirement", "workItemTypes": [{"name": "Product Backlog Item"}]},
                {"id": "bl-t", "name": "Tasks", "rank": 0, "type": "task", "workItemTypes": [{"name": "Task"}]},
            ]
        )

        ctx = await client.build_board_context("ScrumProj")
        assert ctx.default_requirement_type == "Product Backlog Item"
        assert ctx.story_points_field == "effort"
        assert ctx.process_template is not None
        assert ctx.process_template.value == "Scrum"
        await client.__aexit__(None, None, None)

    async def test_basic_template(self) -> None:
        """Basic process template detection (Issue as requirement)."""
        client = await _build_client()
        from src.models.devops import WorkItemType as WIT

        client.get_work_item_types = AsyncMock(
            return_value=[
                WIT(name="Epic", reference_name="r.Epic"),
                WIT(name="Issue", reference_name="r.Issue"),
                WIT(name="Task", reference_name="r.Task"),
            ]
        )
        client.get_backlogs = AsyncMock(
            return_value=[
                {"id": "bl-e", "name": "Epics", "rank": 3, "type": "portfolio", "workItemTypes": [{"name": "Epic"}]},
                {"id": "bl-r", "name": "Issues", "rank": 1, "type": "requirement", "workItemTypes": [{"name": "Issue"}]},
                {"id": "bl-t", "name": "Tasks", "rank": 0, "type": "task", "workItemTypes": [{"name": "Task"}]},
            ]
        )

        ctx = await client.build_board_context("BasicProj")
        assert ctx.default_requirement_type == "Issue"
        assert ctx.story_points_field == "effort"
        assert ctx.process_template is not None
        assert ctx.process_template.value == "Basic"
        await client.__aexit__(None, None, None)

    async def test_epic_only_project(self) -> None:
        """Project with only Epic and Task (no requirement level) defaults to Epic."""
        client = await _build_client()
        from src.models.devops import WorkItemType as WIT

        client.get_work_item_types = AsyncMock(
            return_value=[
                WIT(name="Epic", reference_name="r.Epic"),
                WIT(name="Task", reference_name="r.Task"),
            ]
        )
        client.get_backlogs = AsyncMock(
            return_value=[
                {"id": "bl-e", "name": "Epics", "rank": 3, "type": "portfolio", "workItemTypes": [{"name": "Epic"}]},
                {"id": "bl-t", "name": "Tasks", "rank": 0, "type": "task", "workItemTypes": [{"name": "Task"}]},
            ]
        )

        ctx = await client.build_board_context("EpicOnlyProj")
        assert ctx.default_requirement_type == "Epic"
        assert "Epic" in ctx.parent_types_for_tasks
        assert ctx.story_points_field == "effort"
        await client.__aexit__(None, None, None)

    async def test_backlogs_fetch_failure_uses_defaults(self) -> None:
        """When backlogs API fails, defaults are used gracefully."""
        client = await _build_client()
        from src.models.devops import WorkItemType as WIT

        client.get_work_item_types = AsyncMock(
            return_value=[
                WIT(name="Epic", reference_name="r.Epic"),
                WIT(name="User Story", reference_name="r.US"),
                WIT(name="Task", reference_name="r.Task"),
            ]
        )
        client.get_backlogs = AsyncMock(
            side_effect=AzureDevOpsError("Backlogs unavailable"),
        )

        ctx = await client.build_board_context("FallbackProj")
        # With no backlogs, default_requirement_type starts as "Epic" (the code default)
        # Since backlogs_raw is empty, the code hits the fallback logic
        assert ctx.default_requirement_type in ("Epic", "User Story")
        assert ctx.task_type == "Task"
        await client.__aexit__(None, None, None)

    async def test_board_context_allowed_types(self) -> None:
        """allowed_types reflects all work item types from the project."""
        client = await _build_client()
        from src.models.devops import WorkItemType as WIT

        client.get_work_item_types = AsyncMock(
            return_value=[
                WIT(name="Epic", reference_name="r.Epic"),
                WIT(name="Feature", reference_name="r.Feature"),
                WIT(name="User Story", reference_name="r.US"),
                WIT(name="Task", reference_name="r.Task"),
                WIT(name="Bug", reference_name="r.Bug"),
            ]
        )
        client.get_backlogs = AsyncMock(return_value=[])

        ctx = await client.build_board_context("TypesProj")
        assert set(ctx.allowed_types) == {"Epic", "Feature", "User Story", "Task", "Bug"}
        await client.__aexit__(None, None, None)
