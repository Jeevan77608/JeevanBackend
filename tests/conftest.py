"""Root test fixtures for the Azure DevOps MCP Server backend."""

from __future__ import annotations

import base64
import os
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.settings import Settings, clear_settings_cache
from src.mcp_tools.base import ToolContext
from src.models.user import SessionContext, TokenInfo, UserProfile
from src.models.work_items import DraftCollection, DraftStatus, DraftWorkItem
from src.session.encryption import EncryptionService
from src.session.memory_store import InMemoryStore


# =============================================================================
# Settings Fixtures
# =============================================================================


@pytest.fixture()
def _mock_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Set minimal environment variables for Settings to construct."""
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret-key-must-be-32-chars!")
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("REDIS_ENABLED", "false")
    monkeypatch.setenv("CHAT_SAVE_ENABLED", "false")
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("DEFAULT_PUSH_TAG", "@AIGenerated")
    yield
    clear_settings_cache()


@pytest.fixture()
def mock_settings(_mock_env: None) -> Settings:
    """Return a Settings instance with test defaults."""
    return Settings()


# =============================================================================
# Encryption Fixtures
# =============================================================================


@pytest.fixture()
def encryption_key_b64() -> str:
    """A valid base64-encoded 32-byte key."""
    return base64.b64encode(os.urandom(32)).decode()


@pytest.fixture()
def encryption_service(encryption_key_b64: str) -> EncryptionService:
    """Return an EncryptionService with a random key."""
    return EncryptionService(encryption_key_b64)


# =============================================================================
# Session Store Fixtures
# =============================================================================


@pytest.fixture()
async def memory_store() -> AsyncGenerator[InMemoryStore, None]:
    """Initialised InMemoryStore — cleanup task is cancelled on teardown."""
    store = InMemoryStore()
    await store.initialize()
    yield store
    await store.close()


# =============================================================================
# User / Token / Session Fixtures
# =============================================================================


@pytest.fixture()
def sample_user() -> UserProfile:
    return UserProfile(
        id="user-123",
        display_name="Test User",
        email="test@example.com",
        user_principal_name="test@example.com",
        tenant_id="00000000-0000-0000-0000-000000000000",
    )


@pytest.fixture()
def sample_token_info() -> TokenInfo:
    return TokenInfo(
        access_token="eyJ0eXAi.test-token",
        refresh_token="refresh-token-value",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        scope="openid profile offline_access",
    )


@pytest.fixture()
def sample_session_context(sample_user: UserProfile, sample_token_info: TokenInfo) -> SessionContext:
    return SessionContext(
        session_id="sess-abc-123",
        user=sample_user,
        token_info=sample_token_info,
        selected_org="https://dev.azure.com/TestOrg",
        selected_project="TestProject",
    )


# =============================================================================
# Tool Context Fixtures
# =============================================================================


@pytest.fixture()
def tool_context() -> ToolContext:
    return ToolContext(
        session_id="sess-test-001",
        user_id="user-123",
        organization_url="https://dev.azure.com/TestOrg",
        project_id="TestProject",
        project_name="Test Project",
        access_token="test-access-token",
    )


# =============================================================================
# Mock Client Fixtures
# =============================================================================


@pytest.fixture()
def mock_devops_client() -> AsyncMock:
    """Mock AzureDevOpsClient returning predictable data."""
    client = AsyncMock()
    client.list_organizations.return_value = [
        {"id": "org-1", "name": "TestOrg", "url": "https://dev.azure.com/TestOrg"},
    ]
    client.list_projects.return_value = [
        {"id": "proj-1", "name": "TestProject", "description": "A test project"},
    ]
    client.get_work_item.return_value = {
        "id": 1,
        "fields": {
            "System.Title": "Test Item",
            "System.WorkItemType": "User Story",
            "System.State": "New",
        },
    }
    client.create_work_item.return_value = {
        "id": 42,
        "url": "https://dev.azure.com/TestOrg/_workitems/edit/42",
    }
    return client


@pytest.fixture()
def mock_openai_client() -> AsyncMock:
    """Mock AzureOpenAIClient returning predictable data."""
    client = AsyncMock()
    client.generate_user_stories.return_value = [
        DraftWorkItem(
            draft_id="draft-001",
            work_item_type="User Story",
            title="Generated Story",
            description="A test story",
            status=DraftStatus.DRAFT,
        ),
    ]
    client.generate_tasks.return_value = [
        DraftWorkItem(
            draft_id="draft-task-001",
            work_item_type="Task",
            title="Generated Task",
            parent_draft_id="draft-001",
            status=DraftStatus.DRAFT,
        ),
    ]
    return client


# =============================================================================
# Draft Fixtures
# =============================================================================


@pytest.fixture()
def sample_draft() -> DraftWorkItem:
    return DraftWorkItem(
        draft_id="draft-001",
        work_item_type="User Story",
        title="As a user I want to login",
        description="Login functionality",
        priority=2,
        story_points=5.0,
        tags=["auth", "mvp"],
    )


@pytest.fixture()
def sample_draft_collection(sample_draft: DraftWorkItem) -> DraftCollection:
    return DraftCollection(
        session_id="sess-test-001",
        project_id="TestProject",
        items=[sample_draft],
    )
