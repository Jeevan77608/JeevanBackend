"""Unit tests for the Azure OpenAI client used for work item generation.

Tests cover:
- Exception hierarchy (OpenAIError, GenerationError, RateLimitError, etc.)
- Token estimation and context truncation utilities
- PromptTemplate enum and SYSTEM_PROMPTS dict
- Dynamic prompt builders (user story, task, bug)
- GenerationResult dataclass
- AzureOpenAIClient init, context manager, _get_client
- _generate: success, rate limit, content filter, context length, non-200, JSON parse
- _parse_work_items: story_points, effort, repro_steps, severity, empty
- generate_user_stories, generate_tasks, generate_bug_report, refine_work_item
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.clients.openai import (
    CHARS_PER_TOKEN,
    MODEL_CONTEXT_WINDOW,
    SYSTEM_PROMPTS,
    AzureOpenAIClient,
    ContentFilterError,
    ContextLengthError,
    GenerationError,
    GenerationResult,
    OpenAIError,
    PromptTemplate,
    RateLimitError,
    build_dynamic_bug_prompt,
    build_dynamic_task_prompt,
    build_dynamic_user_story_prompt,
    estimate_tokens,
    truncate_context,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_settings(
    *,
    endpoint: str = "https://test.openai.azure.com/",
    api_key: str = "test-key-123",
    deployment_name: str = "gpt-4",
    api_version: str = "2024-02-01",
    max_tokens: int = 4000,
    temperature: float = 0.7,
) -> MagicMock:
    """Create a mock AzureOpenAISettings."""
    settings = MagicMock()
    settings.endpoint = endpoint
    settings.api_key.get_secret_value.return_value = api_key
    settings.deployment_name = deployment_name
    settings.api_version = api_version
    settings.max_tokens = max_tokens
    settings.temperature = temperature
    return settings


def _make_openai_response(
    status_code: int = 200,
    content: str = '{"work_items": []}',
    usage: dict[str, int] | None = None,
    model: str = "gpt-4",
    headers: dict[str, str] | None = None,
    raw_data: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock httpx.Response for the OpenAI chat completions endpoint."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = httpx.Headers(headers or {})
    resp.text = json.dumps(raw_data) if raw_data else f'{{"choices": [{{"message": {{"content": {json.dumps(content)}}}}}], "usage": {json.dumps(usage or {})}, "model": "{model}"}}'

    if raw_data:
        resp.json.return_value = raw_data
    else:
        resp.json.return_value = {
            "choices": [{"message": {"content": content}}],
            "usage": usage or {},
            "model": model,
        }
    return resp


def _make_board_context(
    *,
    default_requirement_type: str = "User Story",
    epic_type: str | None = "Epic",
    task_type: str = "Task",
    bug_type: str = "Bug",
    feature_type: str | None = "Feature",
    allowed_types: list[str] | None = None,
    hierarchy_levels: list[dict[str, Any]] | None = None,
    parent_types_for_tasks: list[str] | None = None,
    story_points_field: str = "story_points",
    process_template: str | None = None,
) -> MagicMock:
    """Create a mock BoardContext."""
    ctx = MagicMock()
    ctx.default_requirement_type = default_requirement_type
    ctx.epic_type = epic_type
    ctx.task_type = task_type
    ctx.bug_type = bug_type
    ctx.feature_type = feature_type
    ctx.allowed_types = allowed_types or ["Epic", "User Story", "Task", "Bug"]
    ctx.hierarchy_levels = hierarchy_levels or []
    ctx.parent_types_for_tasks = parent_types_for_tasks or ["User Story"]
    ctx.story_points_field = story_points_field
    ctx.process_template = process_template
    ctx.get_allowed_types_string.return_value = ", ".join(
        ctx.allowed_types
    )
    ctx.get_hierarchy_description.return_value = "Epic -> User Story -> Task"
    return ctx


async def _build_client(
    settings: MagicMock | None = None,
) -> AzureOpenAIClient:
    """Build and enter an AzureOpenAIClient with a mocked httpx.AsyncClient."""
    settings = settings or _make_settings()
    client = AzureOpenAIClient(settings)
    await client.__aenter__()
    return client


# =============================================================================
# Exception hierarchy
# =============================================================================


class TestExceptionHierarchy:
    """Verify custom OpenAI exception classes and their attributes."""

    def test_openai_error_base(self) -> None:
        """OpenAIError is the base for all OpenAI client exceptions."""
        err = OpenAIError("base error")
        assert str(err) == "base error"
        assert isinstance(err, Exception)

    def test_generation_error(self) -> None:
        """GenerationError is a subclass of OpenAIError."""
        err = GenerationError("gen failed")
        assert isinstance(err, OpenAIError)

    def test_rate_limit_error(self) -> None:
        """RateLimitError carries retry_after and is a subclass of OpenAIError."""
        err = RateLimitError(retry_after=30.0)
        assert isinstance(err, OpenAIError)
        assert err.retry_after == 30.0
        assert "Rate limit" in str(err)

    def test_rate_limit_error_default(self) -> None:
        """Default retry_after is 60 seconds."""
        err = RateLimitError()
        assert err.retry_after == 60.0

    def test_content_filter_error(self) -> None:
        """ContentFilterError is a subclass of OpenAIError."""
        err = ContentFilterError("blocked")
        assert isinstance(err, OpenAIError)

    def test_context_length_error(self) -> None:
        """ContextLengthError stores estimated and max token counts."""
        err = ContextLengthError(estimated_tokens=200_000, max_tokens=128_000)
        assert isinstance(err, OpenAIError)
        assert err.estimated_tokens == 200_000
        assert err.max_tokens == 128_000
        assert "200,000" in str(err)
        assert "128,000" in str(err)


# =============================================================================
# Token estimation and truncation
# =============================================================================


class TestTokenEstimation:
    """Tests for estimate_tokens and truncate_context utilities."""

    def test_estimate_tokens_basic(self) -> None:
        """Estimates ~4 characters per token."""
        text = "a" * 400
        assert estimate_tokens(text) == 100

    def test_estimate_tokens_empty(self) -> None:
        """Empty string returns 0 tokens."""
        assert estimate_tokens("") == 0

    @pytest.mark.parametrize(
        "length,expected",
        [
            (4, 1),
            (7, 1),  # integer division: 7 // 4 = 1
            (8, 2),
            (0, 0),
        ],
    )
    def test_estimate_tokens_parametrized(self, length: int, expected: int) -> None:
        """Parametrized check of token estimation."""
        assert estimate_tokens("x" * length) == expected

    def test_truncate_context_no_truncation(self) -> None:
        """Text shorter than max_chars is returned unchanged."""
        text = "Hello, world!"
        result = truncate_context(text, max_chars=100, label="test")
        assert result == text

    def test_truncate_context_exact_limit(self) -> None:
        """Text at exactly max_chars is returned unchanged."""
        text = "A" * 100
        result = truncate_context(text, max_chars=100)
        assert result == text

    def test_truncate_context_truncation_at_newline(self) -> None:
        """Truncation prefers cutting at a newline boundary."""
        # Build text that exceeds limit, with a newline at ~90% mark
        text = "A" * 85 + "\n" + "B" * 20  # 106 chars total
        result = truncate_context(text, max_chars=100, label="ctx")
        # Should cut at the newline (position 85) since 85 > 100*0.8=80
        assert result.startswith("A" * 85)
        assert "[WARNING: ctx was truncated" in result
        assert "B" * 20 not in result

    def test_truncate_context_short_content_no_newline(self) -> None:
        """When no newline is found past 80% mark, truncation is at max_chars."""
        text = "X" * 200  # No newlines
        result = truncate_context(text, max_chars=100, label="data")
        assert result.startswith("X" * 100)
        assert "[WARNING: data was truncated" in result


# =============================================================================
# PromptTemplate and SYSTEM_PROMPTS
# =============================================================================


class TestPromptTemplatesAndSystemPrompts:
    """Verify PromptTemplate enum values and SYSTEM_PROMPTS completeness."""

    def test_prompt_template_values(self) -> None:
        """All expected prompt template values exist."""
        assert PromptTemplate.USER_STORY.value == "user_story"
        assert PromptTemplate.TASK.value == "task"
        assert PromptTemplate.BUG.value == "bug"
        assert PromptTemplate.FEATURE.value == "feature"
        assert PromptTemplate.EPIC.value == "epic"

    def test_system_prompts_has_all_keys(self) -> None:
        """SYSTEM_PROMPTS dict has an entry for every PromptTemplate member."""
        for template in PromptTemplate:
            assert template in SYSTEM_PROMPTS, f"Missing SYSTEM_PROMPTS entry for {template}"

    def test_system_prompts_are_nonempty_strings(self) -> None:
        """Each system prompt is a non-empty string."""
        for template, prompt in SYSTEM_PROMPTS.items():
            assert isinstance(prompt, str), f"{template} prompt is not a string"
            assert len(prompt) > 50, f"{template} prompt is suspiciously short"


# =============================================================================
# Dynamic prompt builders
# =============================================================================


class TestDynamicPromptBuilders:
    """Tests for build_dynamic_user_story_prompt, build_dynamic_task_prompt, build_dynamic_bug_prompt."""

    # --- User Story Prompt ---

    def test_user_story_prompt_none_context(self) -> None:
        """With None board_context, returns the static default USER_STORY prompt."""
        result = build_dynamic_user_story_prompt(None)
        assert result == SYSTEM_PROMPTS[PromptTemplate.USER_STORY]

    def test_user_story_prompt_with_context(self) -> None:
        """With board_context, returns a dynamic prompt mentioning the project types."""
        ctx = _make_board_context(
            default_requirement_type="Product Backlog Item",
            story_points_field="effort",
        )
        result = build_dynamic_user_story_prompt(ctx)
        assert "Product Backlog Item" in result
        assert "effort" in result
        assert "PROJECT BOARD STRUCTURE" in result
        # Should NOT be the static default
        assert result != SYSTEM_PROMPTS[PromptTemplate.USER_STORY]

    def test_user_story_prompt_agile_uses_story_points(self) -> None:
        """Agile project uses story_points in the prompt."""
        ctx = _make_board_context(
            default_requirement_type="User Story",
            story_points_field="story_points",
        )
        result = build_dynamic_user_story_prompt(ctx)
        assert "story_points" in result

    # --- Task Prompt ---

    def test_task_prompt_none_context(self) -> None:
        """With None board_context, returns the static default TASK prompt."""
        result = build_dynamic_task_prompt(None)
        assert result == SYSTEM_PROMPTS[PromptTemplate.TASK]

    def test_task_prompt_with_context(self) -> None:
        """With board_context, returns a dynamic prompt with task and parent types."""
        ctx = _make_board_context(
            task_type="Task",
            parent_types_for_tasks=["User Story", "Bug"],
        )
        result = build_dynamic_task_prompt(ctx)
        assert "Task" in result
        assert "User Story" in result
        assert "Bug" in result
        assert "PROJECT BOARD STRUCTURE" in result

    # --- Bug Prompt ---

    def test_bug_prompt_none_context(self) -> None:
        """With None board_context, returns the static default BUG prompt."""
        result = build_dynamic_bug_prompt(None)
        assert result == SYSTEM_PROMPTS[PromptTemplate.BUG]

    def test_bug_prompt_with_context(self) -> None:
        """With board_context, returns a dynamic prompt mentioning the bug type."""
        ctx = _make_board_context(bug_type="Defect")
        result = build_dynamic_bug_prompt(ctx)
        assert "Defect" in result
        assert "PROJECT BOARD STRUCTURE" in result


# =============================================================================
# GenerationResult dataclass
# =============================================================================


class TestGenerationResult:
    """Verify GenerationResult dataclass."""

    def test_construction(self) -> None:
        """GenerationResult can be created with all required fields."""
        from src.models.work_items import DraftWorkItem

        items = [
            DraftWorkItem(draft_id="d1", work_item_type="User Story", title="S1"),
        ]
        result = GenerationResult(
            work_items=items,
            prompt_tokens=100,
            completion_tokens=50,
            model="gpt-4",
        )
        assert len(result.work_items) == 1
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.model == "gpt-4"


# =============================================================================
# AzureOpenAIClient — Init & Context Manager
# =============================================================================


class TestClientInitAndContextManager:
    """Client initialization and async context manager behaviour."""

    def test_init_stores_settings(self) -> None:
        """Client extracts endpoint, api_key, deployment, api_version from settings."""
        settings = _make_settings(
            endpoint="https://my-oai.openai.azure.com/",
            api_key="secret-key",
            deployment_name="gpt-4-turbo",
            api_version="2024-06-01",
        )
        client = AzureOpenAIClient(settings)
        assert client._endpoint == "https://my-oai.openai.azure.com/"
        assert client._api_key == "secret-key"
        assert client._deployment == "gpt-4-turbo"
        assert client._api_version == "2024-06-01"

    async def test_context_manager_creates_and_closes_client(self) -> None:
        """__aenter__ creates httpx client; __aexit__ closes it."""
        settings = _make_settings()
        client = AzureOpenAIClient(settings)
        assert client._client is None

        async with client as c:
            assert c._client is not None
            assert isinstance(c._client, httpx.AsyncClient)

        assert client._client is None

    def test_get_client_raises_without_context(self) -> None:
        """_get_client raises RuntimeError when not in context manager."""
        settings = _make_settings()
        client = AzureOpenAIClient(settings)
        with pytest.raises(RuntimeError, match="not initialized"):
            client._get_client()


# =============================================================================
# _generate — core generation method
# =============================================================================


class TestGenerate:
    """Tests for _generate: success, rate limit, content filter, context length, errors."""

    async def test_generate_success(self) -> None:
        """Successful 200 response is parsed into content + usage + model."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.post.return_value = _make_openai_response(
            status_code=200,
            content='{"work_items": [{"title": "Test"}]}',
            usage={"prompt_tokens": 50, "completion_tokens": 30},
            model="gpt-4-turbo",
        )
        client._client = mock_http

        result = await client._generate("system prompt", "user prompt")
        assert result["content"] == {"work_items": [{"title": "Test"}]}
        assert result["usage"]["prompt_tokens"] == 50
        assert result["model"] == "gpt-4-turbo"
        await client.__aexit__(None, None, None)

    async def test_generate_rate_limit_429(self) -> None:
        """429 response raises RateLimitError with Retry-After value."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.post.return_value = _make_openai_response(
            status_code=429,
            headers={"Retry-After": "20"},
            raw_data={},
        )
        client._client = mock_http

        with pytest.raises(RateLimitError) as exc_info:
            await client._generate("sys", "user")
        assert exc_info.value.retry_after == 20.0
        await client.__aexit__(None, None, None)

    async def test_generate_content_filter_400(self) -> None:
        """400 response with content_filter in body raises ContentFilterError."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.post.return_value = _make_openai_response(
            status_code=400,
            raw_data={"error": {"code": "content_filter", "message": "Blocked"}},
        )
        client._client = mock_http

        with pytest.raises(ContentFilterError):
            await client._generate("sys", "user")
        await client.__aexit__(None, None, None)

    async def test_generate_context_length_exceeded_400(self) -> None:
        """400 response with context_length_exceeded raises ContextLengthError."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.post.return_value = _make_openai_response(
            status_code=400,
            raw_data={"error": {"code": "context_length_exceeded", "message": "Too long"}},
        )
        client._client = mock_http

        with pytest.raises(ContextLengthError):
            await client._generate("sys", "user")
        await client.__aexit__(None, None, None)

    async def test_generate_context_length_pre_validation(self) -> None:
        """Input exceeding model context window raises ContextLengthError before request."""
        settings = _make_settings(max_tokens=4000)
        client = await _build_client(settings=settings)
        mock_http = AsyncMock()
        client._client = mock_http

        # Create prompts that exceed the context window
        huge_prompt = "x" * (MODEL_CONTEXT_WINDOW * CHARS_PER_TOKEN + 1)

        with pytest.raises(ContextLengthError):
            await client._generate("sys", huge_prompt)

        # No HTTP request should have been made
        mock_http.post.assert_not_called()
        await client.__aexit__(None, None, None)

    async def test_generate_non_200_error(self) -> None:
        """Non-200, non-429, non-400 response raises GenerationError."""
        client = await _build_client()
        mock_http = AsyncMock()
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 500
        resp.text = "Internal Server Error"
        resp.headers = httpx.Headers({})
        mock_http.post.return_value = resp
        client._client = mock_http

        with pytest.raises(GenerationError, match="500"):
            await client._generate("sys", "user")
        await client.__aexit__(None, None, None)

    async def test_generate_json_parse_error(self) -> None:
        """Malformed JSON in response raises GenerationError."""
        client = await _build_client()
        mock_http = AsyncMock()
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.headers = httpx.Headers({})
        # Return a response where content is not valid JSON
        resp.json.return_value = {
            "choices": [{"message": {"content": "not valid json {{"}}],
            "usage": {},
            "model": "gpt-4",
        }
        mock_http.post.return_value = resp
        client._client = mock_http

        with pytest.raises(GenerationError, match="parse"):
            await client._generate("sys", "user")
        await client.__aexit__(None, None, None)

    async def test_generate_http_error(self) -> None:
        """httpx.HTTPError during request raises GenerationError."""
        client = await _build_client()
        mock_http = AsyncMock()
        mock_http.post.side_effect = httpx.ConnectError("Connection refused")
        client._client = mock_http

        with pytest.raises(GenerationError, match="HTTP error"):
            await client._generate("sys", "user")
        await client.__aexit__(None, None, None)


# =============================================================================
# _parse_work_items
# =============================================================================


class TestParseWorkItems:
    """Tests for _parse_work_items: various field mappings and edge cases."""

    def _make_client(self) -> AzureOpenAIClient:
        """Create a client instance (no context manager needed for parsing)."""
        return AzureOpenAIClient(_make_settings())

    def test_parse_with_story_points(self) -> None:
        """story_points field is mapped correctly."""
        client = self._make_client()
        data = {
            "work_items": [
                {
                    "work_item_type": "User Story",
                    "title": "Story A",
                    "description": "Desc A",
                    "story_points": 5,
                    "priority": 2,
                },
            ],
        }
        items = client._parse_work_items(data, "User Story")
        assert len(items) == 1
        assert items[0].story_points == 5.0
        assert items[0].title == "Story A"
        assert items[0].priority == 2

    def test_parse_with_effort(self) -> None:
        """effort field is used as story_points when story_points is absent."""
        client = self._make_client()
        data = {
            "work_items": [
                {
                    "title": "PBI",
                    "description": "Desc",
                    "effort": 8,
                },
            ],
        }
        items = client._parse_work_items(data, "Product Backlog Item")
        assert items[0].story_points == 8.0
        assert items[0].work_item_type == "Product Backlog Item"

    def test_parse_with_repro_steps(self) -> None:
        """repro_steps are appended to description for bugs."""
        client = self._make_client()
        data = {
            "work_items": [
                {
                    "title": "Login Bug",
                    "description": "Login fails",
                    "repro_steps": "1. Open app\n2. Click login\n3. Error appears",
                    "severity": 2,
                    "priority": 1,
                },
            ],
        }
        items = client._parse_work_items(data, "Bug")
        assert "Repro Steps" in items[0].description
        assert "1. Open app" in items[0].description
        assert "Login fails" in items[0].description

    def test_parse_repro_steps_no_description(self) -> None:
        """repro_steps works even when description is empty."""
        client = self._make_client()
        data = {
            "work_items": [
                {
                    "title": "No Desc Bug",
                    "description": "",
                    "repro_steps": "1. Step one",
                },
            ],
        }
        items = client._parse_work_items(data, "Bug")
        assert "Repro Steps" in items[0].description
        assert "1. Step one" in items[0].description

    def test_parse_with_severity(self) -> None:
        """severity is stored in custom_fields."""
        client = self._make_client()
        data = {
            "work_items": [
                {
                    "title": "Sev Bug",
                    "severity": 3,
                },
            ],
        }
        items = client._parse_work_items(data, "Bug")
        assert "Microsoft.VSTS.Common.Severity" in items[0].custom_fields
        assert items[0].custom_fields["Microsoft.VSTS.Common.Severity"] == "3"

    def test_parse_empty_items(self) -> None:
        """Empty work_items list returns empty list."""
        client = self._make_client()
        items = client._parse_work_items({"work_items": []}, "Task")
        assert items == []

    def test_parse_missing_work_items_key(self) -> None:
        """Missing work_items key returns empty list."""
        client = self._make_client()
        items = client._parse_work_items({}, "Task")
        assert items == []

    def test_parse_default_type_fallback(self) -> None:
        """When item data lacks work_item_type, falls back to the passed-in default."""
        client = self._make_client()
        data = {
            "work_items": [
                {"title": "No Type"},
            ],
        }
        items = client._parse_work_items(data, "Epic")
        assert items[0].work_item_type == "Epic"

    def test_parse_preserves_item_type_from_data(self) -> None:
        """When item data has work_item_type, it overrides the default."""
        client = self._make_client()
        data = {
            "work_items": [
                {"title": "Override", "work_item_type": "Feature"},
            ],
        }
        items = client._parse_work_items(data, "User Story")
        assert items[0].work_item_type == "Feature"

    def test_parse_no_story_points_no_effort(self) -> None:
        """When neither story_points nor effort is present, story_points is None."""
        client = self._make_client()
        data = {
            "work_items": [
                {"title": "No Points"},
            ],
        }
        items = client._parse_work_items(data, "Task")
        assert items[0].story_points is None


# =============================================================================
# generate_user_stories
# =============================================================================


class TestGenerateUserStories:
    """Tests for generate_user_stories: integration with _generate and _parse_work_items."""

    async def test_generate_user_stories_success(self) -> None:
        """Successful generation returns GenerationResult with parsed work items."""
        client = await _build_client()

        # Mock _generate to return predictable data
        client._generate = AsyncMock(
            return_value={
                "content": {
                    "work_items": [
                        {
                            "work_item_type": "User Story",
                            "title": "Login Feature",
                            "description": "As a user, I want to login",
                            "acceptance_criteria": "Given valid credentials...",
                            "story_points": 5,
                            "priority": 2,
                        },
                    ],
                },
                "usage": {"prompt_tokens": 100, "completion_tokens": 200},
                "model": "gpt-4",
            }
        )

        result = await client.generate_user_stories("Build a login page")
        assert isinstance(result, GenerationResult)
        assert len(result.work_items) == 1
        assert result.work_items[0].title == "Login Feature"
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 200
        assert result.model == "gpt-4"
        await client.__aexit__(None, None, None)

    async def test_generate_user_stories_with_context(self) -> None:
        """Context string is prepended to user prompt."""
        client = await _build_client()
        client._generate = AsyncMock(
            return_value={
                "content": {"work_items": []},
                "usage": {},
                "model": "gpt-4",
            }
        )

        await client.generate_user_stories(
            "Build a dashboard",
            context="Existing Epic: Analytics Platform",
            count=6,
        )

        # Verify the user prompt includes context
        call_args = client._generate.call_args
        user_prompt = call_args.kwargs["user_prompt"]
        assert "Existing Epic: Analytics Platform" in user_prompt
        assert "Build a dashboard" in user_prompt
        assert "Generate exactly 6 work item(s)." in user_prompt
        await client.__aexit__(None, None, None)

    async def test_generate_user_stories_with_board_context(self) -> None:
        """Board context changes the system prompt and default type."""
        client = await _build_client()
        client._generate = AsyncMock(
            return_value={
                "content": {
                    "work_items": [
                        {"title": "PBI Item", "effort": 3},
                    ],
                },
                "usage": {},
                "model": "gpt-4",
            }
        )

        board_ctx = _make_board_context(
            default_requirement_type="Product Backlog Item",
            story_points_field="effort",
        )

        result = await client.generate_user_stories(
            "Build a feature",
            board_context=board_ctx,
        )

        # Default type should be Product Backlog Item
        assert result.work_items[0].work_item_type == "Product Backlog Item"
        # System prompt should be dynamic (not the static default)
        call_args = client._generate.call_args
        system_prompt = call_args.kwargs["system_prompt"]
        assert "Product Backlog Item" in system_prompt
        await client.__aexit__(None, None, None)


# =============================================================================
# generate_tasks
# =============================================================================


class TestGenerateTasks:
    """Tests for generate_tasks: with/without board_context and parent_type."""

    async def test_generate_tasks_default(self) -> None:
        """Generates tasks from a user story (default behaviour)."""
        client = await _build_client()
        client._generate = AsyncMock(
            return_value={
                "content": {
                    "work_items": [
                        {
                            "work_item_type": "Task",
                            "title": "Implement API endpoint",
                            "description": "Create REST endpoint",
                            "remaining_work": 4,
                            "activity": "Development",
                        },
                    ],
                },
                "usage": {"prompt_tokens": 80, "completion_tokens": 120},
                "model": "gpt-4",
            }
        )

        result = await client.generate_tasks("As a user, I want to login")
        assert len(result.work_items) == 1
        assert result.work_items[0].work_item_type == "Task"

        # Verify user prompt includes "User Story:" prefix
        call_args = client._generate.call_args
        user_prompt = call_args.kwargs["user_prompt"]
        assert "User Story:" in user_prompt
        await client.__aexit__(None, None, None)

    async def test_generate_tasks_with_board_context(self) -> None:
        """Board context changes the task type and system prompt."""
        client = await _build_client()
        client._generate = AsyncMock(
            return_value={
                "content": {"work_items": [{"title": "Dev Task"}]},
                "usage": {},
                "model": "gpt-4",
            }
        )

        board_ctx = _make_board_context(
            task_type="Task",
            parent_types_for_tasks=["Issue"],
            default_requirement_type="Issue",
        )

        result = await client.generate_tasks(
            "Fix the bug",
            board_context=board_ctx,
        )

        # User prompt should use board_context default requirement type
        call_args = client._generate.call_args
        user_prompt = call_args.kwargs["user_prompt"]
        assert "Issue:" in user_prompt
        await client.__aexit__(None, None, None)

    async def test_generate_tasks_with_parent_type(self) -> None:
        """Explicit parent_type overrides board_context default in user prompt."""
        client = await _build_client()
        client._generate = AsyncMock(
            return_value={
                "content": {"work_items": [{"title": "Epic Task"}]},
                "usage": {},
                "model": "gpt-4",
            }
        )

        board_ctx = _make_board_context(default_requirement_type="User Story")

        result = await client.generate_tasks(
            "Build platform",
            board_context=board_ctx,
            parent_type="Epic",
        )

        call_args = client._generate.call_args
        user_prompt = call_args.kwargs["user_prompt"]
        assert "Epic:" in user_prompt
        # Should NOT say "User Story:"
        assert "User Story:" not in user_prompt
        await client.__aexit__(None, None, None)

    async def test_generate_tasks_with_context(self) -> None:
        """Context string is prepended to user prompt."""
        client = await _build_client()
        client._generate = AsyncMock(
            return_value={
                "content": {"work_items": []},
                "usage": {},
                "model": "gpt-4",
            }
        )

        await client.generate_tasks(
            "Build feature",
            context="Project uses React",
        )

        call_args = client._generate.call_args
        user_prompt = call_args.kwargs["user_prompt"]
        assert "Project uses React" in user_prompt
        await client.__aexit__(None, None, None)


# =============================================================================
# generate_bug_report
# =============================================================================


class TestGenerateBugReport:
    """Tests for generate_bug_report: with/without board_context."""

    async def test_generate_bug_report_default(self) -> None:
        """Generates a bug report with default Bug type."""
        client = await _build_client()
        client._generate = AsyncMock(
            return_value={
                "content": {
                    "work_items": [
                        {
                            "work_item_type": "Bug",
                            "title": "Login crashes on empty password",
                            "description": "Expected: validation error\nActual: crash",
                            "repro_steps": "1. Open login\n2. Leave password empty\n3. Click submit",
                            "severity": 2,
                            "priority": 1,
                        },
                    ],
                },
                "usage": {"prompt_tokens": 60, "completion_tokens": 150},
                "model": "gpt-4",
            }
        )

        result = await client.generate_bug_report("Login crashes when password is empty")
        assert len(result.work_items) == 1
        bug = result.work_items[0]
        assert bug.work_item_type == "Bug"
        assert "Repro Steps" in bug.description
        assert "Microsoft.VSTS.Common.Severity" in bug.custom_fields
        await client.__aexit__(None, None, None)

    async def test_generate_bug_report_with_board_context(self) -> None:
        """Board context changes the bug type in the system prompt."""
        client = await _build_client()
        client._generate = AsyncMock(
            return_value={
                "content": {"work_items": [{"title": "Defect Found"}]},
                "usage": {},
                "model": "gpt-4",
            }
        )

        board_ctx = _make_board_context(bug_type="Defect")

        result = await client.generate_bug_report(
            "Something is broken",
            board_context=board_ctx,
        )

        # The default work_item_type for parsing should be "Defect"
        assert result.work_items[0].work_item_type == "Defect"

        # System prompt should mention Defect
        call_args = client._generate.call_args
        system_prompt = call_args.kwargs["system_prompt"]
        assert "Defect" in system_prompt
        await client.__aexit__(None, None, None)

    async def test_generate_bug_report_with_context(self) -> None:
        """Context string is included in the user prompt."""
        client = await _build_client()
        client._generate = AsyncMock(
            return_value={
                "content": {"work_items": []},
                "usage": {},
                "model": "gpt-4",
            }
        )

        await client.generate_bug_report(
            "Crash on submit",
            context="Production environment, Chrome 120",
        )

        call_args = client._generate.call_args
        user_prompt = call_args.kwargs["user_prompt"]
        assert "Production environment, Chrome 120" in user_prompt
        assert "Crash on submit" in user_prompt
        await client.__aexit__(None, None, None)


# =============================================================================
# refine_work_item
# =============================================================================


class TestRefineWorkItem:
    """Tests for refine_work_item: updates fields based on AI feedback."""

    async def test_refine_updates_fields(self) -> None:
        """Refinement updates title, description, acceptance_criteria, story_points, priority."""
        client = await _build_client()
        client._generate = AsyncMock(
            return_value={
                "content": {
                    "title": "Refined Title",
                    "description": "Refined description",
                    "acceptance_criteria": "Refined AC",
                    "story_points": 8,
                    "priority": 1,
                },
                "usage": {},
                "model": "gpt-4",
            }
        )

        # Use MagicMock because production code accesses work_item.fields dict
        original = MagicMock()
        original.work_item_type = "User Story"
        original.title = "Original Title"
        original.description = "Original desc"
        original.acceptance_criteria = "Original AC"
        original.story_points = 3.0
        original.priority = 3
        original.fields = {}

        refined = await client.refine_work_item(original, "Make it more detailed")
        assert refined.title == "Refined Title"
        assert refined.description == "Refined description"
        assert refined.acceptance_criteria == "Refined AC"
        assert refined.story_points == 8.0
        assert refined.priority == 1
        # Same object reference should be returned
        assert refined is original
        await client.__aexit__(None, None, None)

    async def test_refine_preserves_unchanged_fields(self) -> None:
        """Fields not in the AI response remain unchanged."""
        client = await _build_client()
        client._generate = AsyncMock(
            return_value={
                "content": {
                    "title": "Updated Title Only",
                    # description, acceptance_criteria, story_points, priority not in response
                },
                "usage": {},
                "model": "gpt-4",
            }
        )

        original = MagicMock()
        original.work_item_type = "Bug"
        original.title = "Old Title"
        original.description = "Old desc"
        original.acceptance_criteria = None
        original.priority = 2
        original.story_points = 5.0
        original.fields = {}

        refined = await client.refine_work_item(original, "Just update title")
        assert refined.title == "Updated Title Only"
        assert refined.description == "Old desc"
        assert refined.priority == 2
        assert refined.story_points == 5.0
        await client.__aexit__(None, None, None)

    async def test_refine_null_story_points_preserved(self) -> None:
        """When AI returns null for story_points, the original value is kept."""
        client = await _build_client()
        client._generate = AsyncMock(
            return_value={
                "content": {
                    "title": "Same",
                    "story_points": None,
                    "priority": None,
                },
                "usage": {},
                "model": "gpt-4",
            }
        )

        original = MagicMock()
        original.work_item_type = "Task"
        original.title = "Task"
        original.description = ""
        original.acceptance_criteria = None
        original.story_points = 3.0
        original.priority = 2
        original.fields = {}

        refined = await client.refine_work_item(original, "No changes needed")
        # None values should not overwrite existing values (code checks for is not None)
        assert refined.story_points == 3.0
        assert refined.priority == 2
        await client.__aexit__(None, None, None)

    async def test_refine_system_prompt_includes_type(self) -> None:
        """System prompt for refinement includes the work item type."""
        client = await _build_client()
        client._generate = AsyncMock(
            return_value={
                "content": {"title": "T"},
                "usage": {},
                "model": "gpt-4",
            }
        )

        item = MagicMock()
        item.work_item_type = "Epic"
        item.title = "E"
        item.description = ""
        item.acceptance_criteria = None
        item.story_points = None
        item.priority = None
        item.fields = {}

        await client.refine_work_item(item, "Feedback")
        call_args = client._generate.call_args
        system_prompt = call_args.kwargs["system_prompt"]
        assert "Epic" in system_prompt
        await client.__aexit__(None, None, None)

    async def test_refine_user_prompt_includes_all_fields(self) -> None:
        """User prompt sent to AI includes all current field values."""
        client = await _build_client()
        client._generate = AsyncMock(
            return_value={
                "content": {"title": "T"},
                "usage": {},
                "model": "gpt-4",
            }
        )

        item = MagicMock()
        item.work_item_type = "User Story"
        item.title = "My Story"
        item.description = "Story desc"
        item.acceptance_criteria = "AC text"
        item.story_points = 5.0
        item.priority = 2
        item.fields = {}

        await client.refine_work_item(item, "Add more detail")
        call_args = client._generate.call_args
        user_prompt = call_args.kwargs["user_prompt"]
        assert "My Story" in user_prompt
        assert "Story desc" in user_prompt
        assert "AC text" in user_prompt
        assert "5" in user_prompt
        assert "Add more detail" in user_prompt
        await client.__aexit__(None, None, None)
