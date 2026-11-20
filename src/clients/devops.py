"""
Azure DevOps REST API v7.0 client.

This module provides a comprehensive client for Azure DevOps Services and Server:
- Organizations, Projects, Teams
- Work Items (CRUD, queries, batch operations)
- Boards and Backlogs
- Wiki pages

Features:
- Rate limiting with token bucket algorithm
- Automatic retry with exponential backoff
- Connection pooling
- Support for both cloud and on-premises

References:
- https://learn.microsoft.com/en-us/rest/api/azure/devops/
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Final

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config.settings import AzureDevOpsSettings
from src.models.devops import (
    Board,
    BoardContext,
    Organization,
    Project,
    Wiki,
    WikiPage,
    WorkItem,
    WorkItemApiResponse,
    WorkItemBatchResponse,
    WorkItemQueryResult,
    WorkItemType,
)
from src.models.work_items import DraftWorkItem, PushResult

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class AzureDevOpsError(Exception):
    """Base exception for Azure DevOps API errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class RateLimitError(AzureDevOpsError):
    """Rate limit exceeded."""

    def __init__(self, retry_after: float = 60.0) -> None:
        super().__init__("Rate limit exceeded", status_code=429)
        self.retry_after = retry_after


class AuthenticationError(AzureDevOpsError):
    """Authentication failed."""

    pass


class NotFoundError(AzureDevOpsError):
    """Resource not found."""

    pass


class ValidationError(AzureDevOpsError):
    """Request validation failed."""

    pass


# =============================================================================
# API Response
# =============================================================================


@dataclass
class ApiResponse:
    """Wrapper for API responses."""

    data: Any
    status_code: int
    headers: dict[str, str] = field(default_factory=dict)
    continuation_token: str | None = None


# =============================================================================
# Rate Limiter
# =============================================================================


@dataclass
class TokenBucket:
    """Token bucket rate limiter."""

    rate: float  # tokens per second
    capacity: float  # max tokens
    tokens: float = field(init=False)
    last_update: float = field(init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        self.tokens = self.capacity
        self.last_update = time.monotonic()

    async def acquire(self, tokens: float = 1.0) -> float:
        """
        Acquire tokens, waiting if necessary.

        Returns the time waited.
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            # Calculate wait time
            wait_time = (tokens - self.tokens) / self.rate
            await asyncio.sleep(wait_time)

            self.tokens = 0
            self.last_update = time.monotonic()


# =============================================================================
# Azure DevOps Client
# =============================================================================


class AzureDevOpsClient:
    """
    Azure DevOps REST API v7.0 client.

    Supports both Azure DevOps Services (cloud) and Azure DevOps Server (on-prem).
    Includes rate limiting, retry logic, and connection pooling.

    Usage:
        async with AzureDevOpsClient(settings, access_token) as client:
            orgs = await client.get_organizations()
    """

    # API base URLs
    VSSPS_URL: Final[str] = "https://app.vssps.visualstudio.com"
    DEVOPS_URL: Final[str] = "https://dev.azure.com"

    def __init__(
        self,
        settings: AzureDevOpsSettings,
        access_token: str,
        organization_url: str | None = None,
        use_pat: bool = False,
    ) -> None:
        """
        Initialize Azure DevOps client.

        Args:
            settings: Azure DevOps settings
            access_token: OAuth access token or Personal Access Token (PAT)
            organization_url: Organization URL (optional, can be set per-request)
            use_pat: If True, use Basic auth with PAT instead of Bearer token
        """
        self._settings = settings
        self._access_token = access_token
        self._org_url = organization_url or settings.default_org_url
        self._api_version = settings.api_version
        self._use_pat = use_pat

        # Rate limiter
        self._rate_limiter = TokenBucket(
            rate=settings.requests_per_second,
            capacity=settings.requests_per_second * 2,
        )

        # HTTP client (initialized in __aenter__)
        self._client: httpx.AsyncClient | None = None

        logger.debug(
            "Azure DevOps client initialized",
            extra={
                "org_url": self._org_url,
                "api_version": self._api_version,
                "use_pat": self._use_pat,
            },
        )

    def _get_auth_header(self) -> str:
        """Get authorization header based on auth type."""
        if self._use_pat:
            # PAT uses Basic auth with empty username
            import base64
            encoded = base64.b64encode(f":{self._access_token}".encode()).decode()
            return f"Basic {encoded}"
        return f"Bearer {self._access_token}"

    async def __aenter__(self) -> AzureDevOpsClient:
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._settings.request_timeout),
            follow_redirects=True,
            headers={
                "Authorization": self._get_auth_header(),
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get HTTP client, ensuring it's initialized."""
        if self._client is None:
            raise RuntimeError("Client not initialized. Use async context manager.")
        return self._client

    # =========================================================================
    # HTTP Methods with Rate Limiting and Retry
    # =========================================================================

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
    )
    async def _request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> ApiResponse:
        """Make HTTP request with rate limiting and retry."""
        # Apply rate limiting
        await self._rate_limiter.acquire()

        client = self._get_client()

        # Add API version to params
        params = kwargs.pop("params", {})
        params["api-version"] = self._api_version
        kwargs["params"] = params

        try:
            response = await client.request(method, url, **kwargs)

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", 60))
                raise RateLimitError(retry_after)

            # Handle errors
            if response.status_code == 401:
                raise AuthenticationError("Authentication failed", status_code=401)
            if response.status_code == 404:
                raise NotFoundError("Resource not found", status_code=404)
            if response.status_code >= 400:
                error_data = response.json() if response.content else {}
                raise AzureDevOpsError(
                    error_data.get("message", f"Request failed: {response.status_code}"),
                    status_code=response.status_code,
                )

            # Parse response
            data = response.json() if response.content else None
            continuation_token = response.headers.get("x-ms-continuationtoken")

            return ApiResponse(
                data=data,
                status_code=response.status_code,
                headers=dict(response.headers),
                continuation_token=continuation_token,
            )

        except httpx.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            raise AzureDevOpsError(f"HTTP error: {e}") from e

    async def _get(self, url: str, **kwargs: Any) -> ApiResponse:
        """GET request."""
        return await self._request("GET", url, **kwargs)

    async def _post(self, url: str, **kwargs: Any) -> ApiResponse:
        """POST request."""
        return await self._request("POST", url, **kwargs)

    async def _patch(self, url: str, **kwargs: Any) -> ApiResponse:
        """PATCH request."""
        return await self._request("PATCH", url, **kwargs)

    async def _delete(self, url: str, **kwargs: Any) -> ApiResponse:
        """DELETE request."""
        return await self._request("DELETE", url, **kwargs)

    # =========================================================================
    # Organization & Project APIs
    # =========================================================================

    async def get_organizations(self) -> list[Organization]:
        """
        Get all organizations the user has access to.

        Returns:
            List of organizations
        """
        url = f"{self.VSSPS_URL}/_apis/accounts"
        response = await self._get(url, params={"memberId": "me"})

        orgs = []
        for org_data in response.data.get("value", []):
            orgs.append(
                Organization(
                    id=org_data["accountId"],
                    name=org_data["accountName"],
                    url=f"{self.DEVOPS_URL}/{org_data['accountName']}",
                )
            )
        return orgs

    async def get_projects(self, org_url: str | None = None) -> list[Project]:
        """
        Get all projects in an organization.

        Args:
            org_url: Organization URL (uses default if not provided)

        Returns:
            List of projects
        """
        base_url = org_url or self._org_url
        if not base_url:
            raise ValueError("Organization URL required")

        url = f"{base_url}/_apis/projects"
        response = await self._get(url)

        projects = []
        for proj_data in response.data.get("value", []):
            projects.append(
                Project(
                    id=proj_data["id"],
                    name=proj_data["name"],
                    description=proj_data.get("description"),
                    url=proj_data.get("url", ""),
                    state=proj_data.get("state", "wellFormed"),
                )
            )
        return projects

    async def get_project(
        self,
        project_id: str,
        org_url: str | None = None,
    ) -> Project:
        """Get a specific project by ID or name."""
        base_url = org_url or self._org_url
        if not base_url:
            raise ValueError("Organization URL required")

        url = f"{base_url}/_apis/projects/{project_id}"
        response = await self._get(url)

        data = response.data
        return Project(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            url=data.get("url", ""),
            state=data.get("state", "wellFormed"),
        )

    # =========================================================================
    # Work Item APIs
    # =========================================================================

    async def get_work_item_types(
        self,
        project: str,
        org_url: str | None = None,
    ) -> list[WorkItemType]:
        """Get available work item types for a project."""
        base_url = org_url or self._org_url
        if not base_url:
            raise ValueError("Organization URL required")

        url = f"{base_url}/{project}/_apis/wit/workitemtypes"
        response = await self._get(url)

        types = []
        for type_data in response.data.get("value", []):
            types.append(
                WorkItemType(
                    name=type_data["name"],
                    reference_name=type_data["referenceName"],
                    description=type_data.get("description"),
                    icon_url=type_data.get("icon", {}).get("url"),
                    color=type_data.get("color"),
                )
            )
        return types

    async def get_work_item(
        self,
        work_item_id: int,
        project: str | None = None,
        org_url: str | None = None,
        expand: str = "All",
    ) -> WorkItem:
        """
        Get a work item by ID with full validation.

        Args:
            work_item_id: Work item ID
            project: Project name or ID (optional)
            org_url: Organization URL
            expand: Expansion options (All, Relations, Fields, Links, None)

        Returns:
            Validated WorkItem model
        """
        base_url = org_url or self._org_url
        if not base_url:
            raise ValueError("Organization URL required")

        if project:
            url = f"{base_url}/{project}/_apis/wit/workitems/{work_item_id}"
        else:
            url = f"{base_url}/_apis/wit/workitems/{work_item_id}"

        response = await self._get(url, params={"$expand": expand})

        # Validate response with Pydantic model
        api_response = WorkItemApiResponse.model_validate(response.data)
        return api_response.to_work_item()

    async def get_work_item_raw(
        self,
        work_item_id: int,
        project: str | None = None,
        org_url: str | None = None,
        expand: str = "All",
    ) -> dict[str, Any]:
        """Get a work item by ID (raw response without validation)."""
        base_url = org_url or self._org_url
        if not base_url:
            raise ValueError("Organization URL required")

        if project:
            url = f"{base_url}/{project}/_apis/wit/workitems/{work_item_id}"
        else:
            url = f"{base_url}/_apis/wit/workitems/{work_item_id}"

        response = await self._get(url, params={"$expand": expand})
        return response.data

    async def get_work_items_batch(
        self,
        work_item_ids: list[int],
        project: str | None = None,
        org_url: str | None = None,
        expand: str = "All",
    ) -> list[WorkItem]:
        """
        Get multiple work items by IDs in a single request.

        Args:
            work_item_ids: List of work item IDs (max 200)
            project: Project name or ID (optional)
            org_url: Organization URL
            expand: Expansion options

        Returns:
            List of validated WorkItem models
        """
        if not work_item_ids:
            return []

        # Azure DevOps limits batch to 200 items
        if len(work_item_ids) > 200:
            raise ValueError("Maximum 200 work items per batch request")

        base_url = org_url or self._org_url
        if not base_url:
            raise ValueError("Organization URL required")

        ids_str = ",".join(str(id) for id in work_item_ids)

        if project:
            url = f"{base_url}/{project}/_apis/wit/workitems"
        else:
            url = f"{base_url}/_apis/wit/workitems"

        response = await self._get(
            url,
            params={"ids": ids_str, "$expand": expand}
        )

        # Validate response
        batch_response = WorkItemBatchResponse.model_validate(response.data)
        return batch_response.to_work_items()

    async def query_work_items(
        self,
        project: str,
        wiql: str,
        org_url: str | None = None,
        top: int = 200,
    ) -> list[int]:
        """
        Execute a WIQL query and return work item IDs.

        Args:
            project: Project name or ID
            wiql: WIQL query string
            org_url: Organization URL
            top: Maximum number of results

        Returns:
            List of work item IDs matching the query
        """
        base_url = org_url or self._org_url
        if not base_url:
            raise ValueError("Organization URL required")

        url = f"{base_url}/{project}/_apis/wit/wiql"

        response = await self._post(
            url,
            json={"query": wiql},
            params={"$top": top},
        )

        # Validate response
        query_result = WorkItemQueryResult.model_validate(response.data)
        return query_result.work_item_ids

    async def search_work_items(
        self,
        project: str,
        work_item_type: str | None = None,
        state: str | None = None,
        area_path: str | None = None,
        iteration_path: str | None = None,
        assigned_to: str | None = None,
        title_contains: str | None = None,
        org_url: str | None = None,
        top: int = 50,
    ) -> list[WorkItem]:
        """
        Search for work items with various filters.

        Args:
            project: Project name or ID
            work_item_type: Filter by type (Epic, User Story, Task, Bug, etc.)
            state: Filter by state (New, Active, Closed, etc.)
            area_path: Filter by area path
            iteration_path: Filter by iteration path
            assigned_to: Filter by assigned user
            title_contains: Filter by title containing text
            org_url: Organization URL
            top: Maximum number of results

        Returns:
            List of matching WorkItem models
        """
        # Build WIQL query
        conditions = []

        if work_item_type:
            conditions.append(f"[System.WorkItemType] = '{work_item_type}'")
        if state:
            conditions.append(f"[System.State] = '{state}'")
        if area_path:
            conditions.append(f"[System.AreaPath] UNDER '{area_path}'")
        if iteration_path:
            conditions.append(f"[System.IterationPath] UNDER '{iteration_path}'")
        if assigned_to:
            conditions.append(f"[System.AssignedTo] = '{assigned_to}'")
        if title_contains:
            conditions.append(f"[System.Title] CONTAINS '{title_contains}'")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        wiql = f"""
        SELECT [System.Id]
        FROM WorkItems
        WHERE [System.TeamProject] = '{project}'
        AND {where_clause}
        ORDER BY [System.ChangedDate] DESC
        """

        # Get work item IDs
        work_item_ids = await self.query_work_items(
            project=project,
            wiql=wiql,
            org_url=org_url,
            top=top,
        )

        if not work_item_ids:
            return []

        # Fetch full work items
        return await self.get_work_items_batch(
            work_item_ids=work_item_ids,
            project=project,
            org_url=org_url,
        )

    async def get_work_item_children(
        self,
        work_item_id: int,
        project: str | None = None,
        org_url: str | None = None,
    ) -> list[WorkItem]:
        """
        Get all child work items of a parent work item.

        Args:
            work_item_id: Parent work item ID
            project: Project name or ID
            org_url: Organization URL

        Returns:
            List of child WorkItem models
        """
        # First get the parent to find child IDs
        parent = await self.get_work_item(
            work_item_id=work_item_id,
            project=project,
            org_url=org_url,
        )

        child_ids = parent.child_ids
        if not child_ids:
            return []

        return await self.get_work_items_batch(
            work_item_ids=child_ids,
            project=project,
            org_url=org_url,
        )

    async def get_work_item_hierarchy(
        self,
        work_item_id: int,
        project: str | None = None,
        org_url: str | None = None,
        max_depth: int = 3,
    ) -> dict[str, Any]:
        """
        Get work item with its full hierarchy (parent and children).

        Args:
            work_item_id: Work item ID
            project: Project name or ID
            org_url: Organization URL
            max_depth: Maximum depth to traverse

        Returns:
            Dictionary with work item and its hierarchy
        """
        work_item = await self.get_work_item(
            work_item_id=work_item_id,
            project=project,
            org_url=org_url,
        )

        result = {
            "work_item": work_item,
            "parent": None,
            "children": [],
        }

        # Get parent if exists
        if work_item.parent_id:
            try:
                result["parent"] = await self.get_work_item(
                    work_item_id=work_item.parent_id,
                    project=project,
                    org_url=org_url,
                )
            except NotFoundError:
                pass

        # Get children if depth allows
        if max_depth > 0 and work_item.child_ids:
            result["children"] = await self.get_work_items_batch(
                work_item_ids=work_item.child_ids,
                project=project,
                org_url=org_url,
            )

        return result

    async def create_work_item(
        self,
        project: str,
        work_item_type: str,
        fields: dict[str, Any],
        org_url: str | None = None,
        parent_id: int | None = None,
    ) -> dict[str, Any]:
        """
        Create a new work item, optionally linked to a parent.

        Args:
            project: Project name or ID
            work_item_type: Work item type (e.g., "User Story", "Bug")
            fields: Field values to set
            org_url: Organization URL
            parent_id: If set, link this work item as child of the given parent

        Returns:
            Created work item data
        """
        base_url = org_url or self._org_url
        if not base_url:
            raise ValueError("Organization URL required")

        url = f"{base_url}/{project}/_apis/wit/workitems/${work_item_type}"

        # Convert fields to JSON Patch format
        operations = [
            {"op": "add", "path": f"/fields/{key}", "value": value}
            for key, value in fields.items()
        ]

        # Add parent link if specified (System.LinkTypes.Hierarchy-Reverse = this item's parent)
        if parent_id is not None:
            parent_url = f"{base_url.rstrip('/')}/_apis/wit/workItems/{parent_id}"
            operations.append({
                "op": "add",
                "path": "/relations/-",
                "value": {
                    "rel": "System.LinkTypes.Hierarchy-Reverse",
                    "url": parent_url,
                    "attributes": {"comment": "Parent-child link from push"},
                },
            })

        response = await self._post(
            url,
            json=operations,
            headers={"Content-Type": "application/json-patch+json"},
        )
        return response.data

    async def add_work_item_link(
        self,
        project: str,
        work_item_id: int,
        target_id: int,
        link_type: str = "System.LinkTypes.Hierarchy-Reverse",
        org_url: str | None = None,
    ) -> dict[str, Any]:
        """
        Add a link from work_item_id to target_id (e.g. make target the parent of work_item).
        """
        base_url = org_url or self._org_url
        if not base_url:
            raise ValueError("Organization URL required")

        # PATCH requires project in path: /{org}/{project}/_apis/wit/workitems/{id}
        url = f"{base_url.rstrip('/')}/{project}/_apis/wit/workitems/{work_item_id}"
        target_url = f"{base_url.rstrip('/')}/_apis/wit/workItems/{target_id}"
        operations = [{
            "op": "add",
            "path": "/relations/-",
            "value": {
                "rel": link_type,
                "url": target_url,
                "attributes": {"comment": "Parent-child link"},
            },
        }]

        response = await self._patch(
            url,
            json=operations,
            headers={"Content-Type": "application/json-patch+json"},
        )
        return response.data

    async def update_work_item(
        self,
        work_item_id: int,
        fields: dict[str, Any],
        org_url: str | None = None,
    ) -> dict[str, Any]:
        """Update an existing work item."""
        base_url = org_url or self._org_url
        if not base_url:
            raise ValueError("Organization URL required")

        url = f"{base_url}/_apis/wit/workitems/{work_item_id}"

        # Convert fields to JSON Patch format
        operations = [
            {"op": "add", "path": f"/fields/{key}", "value": value}
            for key, value in fields.items()
        ]

        response = await self._patch(
            url,
            json=operations,
            headers={"Content-Type": "application/json-patch+json"},
        )
        return response.data



    # =========================================================================
    # Board APIs
    # =========================================================================

    async def get_boards(
        self,
        project: str,
        team: str | None = None,
        org_url: str | None = None,
    ) -> list[Board]:
        """Get boards for a project/team."""
        base_url = org_url or self._org_url
        if not base_url:
            raise ValueError("Organization URL required")

        if team:
            url = f"{base_url}/{project}/{team}/_apis/work/boards"
        else:
            url = f"{base_url}/{project}/_apis/work/boards"

        response = await self._get(url)

        boards = []
        for board_data in response.data.get("value", []):
            boards.append(
                Board(
                    id=board_data["id"],
                    name=board_data["name"],
                    url=board_data.get("url", ""),
                )
            )
        return boards

    async def get_board_columns(
        self,
        project: str,
        board_id: str,
        team: str | None = None,
        org_url: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get columns for a board."""
        base_url = org_url or self._org_url
        if not base_url:
            raise ValueError("Organization URL required")

        if team:
            url = f"{base_url}/{project}/{team}/_apis/work/boards/{board_id}/columns"
        else:
            url = f"{base_url}/{project}/_apis/work/boards/{board_id}/columns"

        response = await self._get(url)
        return response.data.get("value", [])

    async def get_backlogs(
        self,
        project: str,
        team: str | None = None,
        org_url: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get backlogs for a project/team."""
        base_url = org_url or self._org_url
        if not base_url:
            raise ValueError("Organization URL required")

        if team:
            url = f"{base_url}/{project}/{team}/_apis/work/backlogs"
        else:
            url = f"{base_url}/{project}/_apis/work/backlogs"

        response = await self._get(url)
        return response.data.get("value", [])

    # =========================================================================
    # Wiki APIs
    # =========================================================================

    async def get_wikis(
        self,
        project: str,
        org_url: str | None = None,
    ) -> list[Wiki]:
        """Get wikis for a project."""
        base_url = org_url or self._org_url
        if not base_url:
            raise ValueError("Organization URL required")

        url = f"{base_url}/{project}/_apis/wiki/wikis"
        response = await self._get(url)

        wikis = []
        for wiki_data in response.data.get("value", []):
            wikis.append(
                Wiki(
                    id=wiki_data["id"],
                    name=wiki_data["name"],
                    project_id=wiki_data.get("projectId", ""),
                    type=wiki_data.get("type", "projectWiki"),
                    url=wiki_data.get("url", ""),
                )
            )
        return wikis

    async def get_wiki_page(
        self,
        project: str,
        wiki_id: str,
        path: str,
        org_url: str | None = None,
        include_content: bool = True,
    ) -> WikiPage:
        """Get a wiki page by path."""
        base_url = org_url or self._org_url
        if not base_url:
            raise ValueError("Organization URL required")

        url = f"{base_url}/{project}/_apis/wiki/wikis/{wiki_id}/pages"
        params = {"path": path}
        if include_content:
            params["includeContent"] = "true"

        response = await self._get(url, params=params)
        data = response.data

        return WikiPage(
            id=data.get("id", ""),
            path=data.get("path", path),
            content=data.get("content"),
            git_item_path=data.get("gitItemPath"),
            order=data.get("order"),
        )

    async def list_wiki_pages(
        self,
        project: str,
        wiki_id: str,
        path: str = "/",
        org_url: str | None = None,
        recursion_level: str = "Full",
    ) -> list[WikiPage]:
        """
        List all wiki pages in a wiki.

        Args:
            project: Project name or ID.
            wiki_id: Wiki identifier or name.
            path: Root path to start listing from (default: "/").
            org_url: Organization URL (optional).
            recursion_level: Recursion level - "None", "OneLevel", or "Full" (default: "Full").

        Returns:
            List of WikiPage objects representing all pages in the wiki.
        """
        base_url = org_url or self._org_url
        if not base_url:
            raise ValueError("Organization URL required")

        url = f"{base_url}/{project}/_apis/wiki/wikis/{wiki_id}/pages"
        params = {
            "path": path,
            "recursionLevel": recursion_level,
        }

        response = await self._get(url, params=params)
        data = response.data

        def flatten_pages(page_data: dict, pages: list[WikiPage]) -> None:
            """Recursively flatten the wiki page tree into a list."""
            # Create WikiPage for current page
            wiki_page = WikiPage(
                id=page_data.get("id", ""),
                path=page_data.get("path", ""),
                content=page_data.get("content"),
                git_item_path=page_data.get("gitItemPath"),
                order=page_data.get("order"),
                url=page_data.get("remoteUrl"),
            )
            pages.append(wiki_page)

            # Recursively process subPages
            sub_pages = page_data.get("subPages", [])
            for sub_page in sub_pages:
                flatten_pages(sub_page, pages)

        pages: list[WikiPage] = []
        flatten_pages(data, pages)
        return pages

    async def create_wiki_page(
        self,
        project: str,
        wiki_id: str,
        path: str,
        content: str,
        org_url: str | None = None,
    ) -> WikiPage:
        """Create or update a wiki page."""
        base_url = org_url or self._org_url
        if not base_url:
            raise ValueError("Organization URL required")

        url = f"{base_url}/{project}/_apis/wiki/wikis/{wiki_id}/pages"
        params = {"path": path}

        response = await self._request(
            "PUT",
            url,
            params=params,
            json={"content": content},
        )
        data = response.data

        return WikiPage(
            id=data.get("id", ""),
            path=data.get("path", path),
            content=content,
            git_item_path=data.get("gitItemPath"),
            order=data.get("order"),
        )

    # =========================================================================
    # Batch Operations
    # =========================================================================

    async def create_work_items_batch(
        self,
        project: str,
        items: list[DraftWorkItem],
        org_url: str | None = None,
    ) -> list[PushResult]:
        """
        Create multiple work items in batch.

        Args:
            project: Project name or ID
            items: List of draft work items to create
            org_url: Organization URL

        Returns:
            List of push results for each item
        """
        results = []
        for item in items:
            try:
                # Build fields from draft properties
                fields: dict[str, Any] = {
                    "System.Title": item.title,
                }
                if item.description:
                    fields["System.Description"] = item.description
                if item.acceptance_criteria:
                    fields["Microsoft.VSTS.Common.AcceptanceCriteria"] = item.acceptance_criteria
                if item.priority:
                    fields["Microsoft.VSTS.Common.Priority"] = item.priority
                if item.story_points:
                    fields["Microsoft.VSTS.Scheduling.StoryPoints"] = item.story_points
                if item.assigned_to:
                    fields["System.AssignedTo"] = item.assigned_to
                if item.tags:
                    fields["System.Tags"] = "; ".join(item.tags)
                if item.area_path:
                    fields["System.AreaPath"] = item.area_path
                if item.iteration_path:
                    fields["System.IterationPath"] = item.iteration_path
                fields.update(item.custom_fields)

                created = await self.create_work_item(
                    project=project,
                    work_item_type=item.work_item_type,
                    fields=fields,
                    org_url=org_url,
                )
                results.append(
                    PushResult(
                        draft_id=item.draft_id,
                        success=True,
                        work_item_id=created.get("id"),
                        work_item_url=created.get("url"),
                    )
                )
            except AzureDevOpsError as e:
                results.append(
                    PushResult(
                        draft_id=item.draft_id,
                        success=False,
                        error_message=str(e),
                    )
                )
        return results

    # =========================================================================
    # Board Context for AI Generation
    # =========================================================================

    async def build_board_context(
        self,
        project: str,
        team: str | None = None,
        org_url: str | None = None,
    ) -> BoardContext:
        """
        Build BoardContext for AI generation from project board structure.

        This fetches work item types and backlog hierarchy to determine:
        - Available work item types (Epic, User Story, Task, Bug, Issue, etc.)
        - Hierarchy (Epic → Feature → Story → Task)
        - Default requirement type (User Story vs Issue vs Product Backlog Item)
        - Parent types for tasks

        Args:
            project: Project name or ID
            team: Team name (optional)
            org_url: Organization URL

        Returns:
            BoardContext with types and hierarchy for AI prompts
        """
        from src.models.devops import ProcessTemplate

        base_url = org_url or self._org_url
        if not base_url:
            raise ValueError("Organization URL required")

        # Fetch work item types
        work_item_types = await self.get_work_item_types(project, org_url=base_url)
        allowed_types = [wit.name for wit in work_item_types]

        # Fetch backlog levels
        try:
            backlogs_raw = await self.get_backlogs(project, team=team, org_url=base_url)
        except Exception as e:
            logger.warning(f"Could not fetch backlogs: {e}. Using defaults.")
            backlogs_raw = []

        # Parse backlog levels
        hierarchy_levels = []
        default_requirement_type = "Epic"  # Default to Epic, not User Story
        parent_types_for_tasks = []
        epic_type = None
        feature_type = None
        task_type = "Task"
        bug_type = "Bug"
        story_points_field = "story_points"

        for backlog in backlogs_raw:
            level_info = {
                "id": backlog.get("id", ""),
                "name": backlog.get("name", ""),
                "rank": backlog.get("rank", 0),
                "work_item_types": [],
                "type": backlog.get("type", ""),  # portfolio, requirement, task
            }

            # Extract work item type names from this backlog level
            wit_array = backlog.get("workItemTypes", [])
            for wit_ref in wit_array:
                wit_name = wit_ref.get("name", "")
                if wit_name:
                    level_info["work_item_types"].append(wit_name)

            hierarchy_levels.append(level_info)

            # Identify special types based on backlog type
            backlog_type = level_info["type"]
            level_types = level_info["work_item_types"]

            if backlog_type == "portfolio" and level_info["rank"] >= 3:
                # High-rank portfolio = Epic level
                if level_types:
                    epic_type = level_types[0]  # Usually "Epic"
            elif backlog_type == "portfolio" and level_info["rank"] == 2:
                # Mid-rank portfolio = Feature level
                if level_types:
                    feature_type = level_types[0]  # Usually "Feature"
            elif backlog_type == "requirement":
                # Requirement level = User Story / Issue / PBI
                if level_types:
                    # First type is the default requirement
                    default_requirement_type = level_types[0]
                    parent_types_for_tasks.extend(level_types)
            elif backlog_type == "task":
                # Task level
                if level_types:
                    task_type = level_types[0]

        # Sort hierarchy by rank (descending) so Epic is first
        hierarchy_levels.sort(key=lambda x: x.get("rank", 0), reverse=True)

        # CRITICAL: If board has only Epic + Task (no User Story/Issue/PBI), derive from actual types
        # Otherwise we wrongly default to "User Story" and UI/generation show wrong types
        allowed_set = set(allowed_types)
        if default_requirement_type not in allowed_set:
            # No requirement level in this project - use top-level type (Epic) as default
            if epic_type and epic_type in allowed_set:
                default_requirement_type = epic_type
                if not parent_types_for_tasks:
                    parent_types_for_tasks = [epic_type]  # Tasks live under Epics
            elif allowed_set:
                # Fallback: use first non-Task type as default (e.g. Epic only)
                non_task = [t for t in allowed_types if t != task_type]
                if non_task:
                    default_requirement_type = non_task[0]
                    if not parent_types_for_tasks:
                        parent_types_for_tasks = [non_task[0]]

        # Ensure default and parent types exist in project
        if default_requirement_type not in allowed_set and allowed_set:
            # Always prefer Epic as fallback, then first non-Task type
            if "Epic" in allowed_set:
                default_requirement_type = "Epic"
            else:
                default_requirement_type = next((t for t in allowed_types if t != task_type), allowed_types[0])

        # Set parent types for tasks
        if not parent_types_for_tasks:
            parent_types_for_tasks = [default_requirement_type] if default_requirement_type in allowed_set else (allowed_types or ["Epic"])
        parent_types_for_tasks = [t for t in parent_types_for_tasks if t in allowed_set]
        if not parent_types_for_tasks and allowed_set:
            # Prefer Epic as parent, then any non-Task type
            if "Epic" in allowed_set:
                parent_types_for_tasks = ["Epic"]
            else:
                parent_types_for_tasks = [t for t in allowed_types if t != task_type][:1] or list(allowed_set)

        # Determine process template hints
        process_template = None
        if default_requirement_type == "Issue":
            process_template = ProcessTemplate.BASIC
            story_points_field = "effort"  # Basic uses Effort instead of Story Points
        elif default_requirement_type == "Product Backlog Item":
            process_template = ProcessTemplate.SCRUM
            story_points_field = "effort"  # Scrum uses Effort
        elif default_requirement_type == "User Story":
            process_template = ProcessTemplate.AGILE
            story_points_field = "story_points"
        elif default_requirement_type == "Epic":
            story_points_field = "effort"  # Epic-level often uses Effort

        return BoardContext(
            project_id=project,
            process_template=process_template,
            allowed_types=allowed_types,
            hierarchy_levels=hierarchy_levels,
            default_requirement_type=default_requirement_type,
            parent_types_for_tasks=parent_types_for_tasks,
            epic_type=epic_type,
            task_type=task_type,
            bug_type=bug_type,
            feature_type=feature_type,
            story_points_field=story_points_field,
        )
