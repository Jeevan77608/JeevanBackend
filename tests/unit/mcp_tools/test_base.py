"""Tests for src/mcp_tools/base.py — ToolStatus, ToolResult, ToolContext, BaseTool."""

from __future__ import annotations

from typing import Any

import pytest

from src.mcp_tools.base import (
    BaseTool,
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    ToolStatus,
)


# =============================================================================
# Concrete subclass used by BaseTool tests
# =============================================================================


class _DummyTool(BaseTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="dummy",
            description="Test tool",
            parameters=[
                ToolParameter(
                    name="required_p",
                    type="string",
                    description="req",
                    required=True,
                ),
                ToolParameter(
                    name="optional_p",
                    type="string",
                    description="opt",
                    required=False,
                    default="default_val",
                ),
            ],
        )

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        return ToolResult.success()


# =============================================================================
# ToolStatus
# =============================================================================


class TestToolStatus:
    """Tests for the ToolStatus enum."""

    def test_success_value(self) -> None:
        assert ToolStatus.SUCCESS == "success"

    def test_error_value(self) -> None:
        assert ToolStatus.ERROR == "error"

    def test_partial_value(self) -> None:
        assert ToolStatus.PARTIAL == "partial"

    def test_all_members_present(self) -> None:
        names = {s.name for s in ToolStatus}
        assert names == {"SUCCESS", "ERROR", "PARTIAL"}


# =============================================================================
# ToolResult
# =============================================================================


class TestToolResult:
    """Tests for the ToolResult dataclass factory methods."""

    def test_success_creates_success_status(self) -> None:
        result = ToolResult.success()
        assert result.status == ToolStatus.SUCCESS

    def test_success_with_data_and_message(self) -> None:
        result = ToolResult.success(data={"count": 5}, message="Found 5 items")
        assert result.data == {"count": 5}
        assert result.message == "Found 5 items"

    def test_success_with_metadata_kwargs(self) -> None:
        result = ToolResult.success(update_context=True, page=2)
        assert result.metadata["update_context"] is True
        assert result.metadata["page"] == 2

    def test_success_defaults(self) -> None:
        result = ToolResult.success()
        assert result.data is None
        assert result.message is None
        assert result.error is None
        assert result.metadata == {}

    def test_failure_creates_error_status(self) -> None:
        result = ToolResult.failure("something went wrong")
        assert result.status == ToolStatus.ERROR

    def test_failure_stores_error_message(self) -> None:
        result = ToolResult.failure("bad input")
        assert result.error == "bad input"

    def test_failure_with_data(self) -> None:
        result = ToolResult.failure("partial data", data={"partial": True})
        assert result.data == {"partial": True}

    def test_failure_with_metadata(self) -> None:
        result = ToolResult.failure("err", retry=True)
        assert result.metadata["retry"] is True

    def test_partial_creates_partial_status(self) -> None:
        result = ToolResult.partial(data={"ok": 3, "fail": 1}, message="3/4 succeeded")
        assert result.status == ToolStatus.PARTIAL

    def test_partial_stores_data_and_message(self) -> None:
        result = ToolResult.partial(data=[1, 2], message="partial result")
        assert result.data == [1, 2]
        assert result.message == "partial result"

    def test_partial_with_metadata(self) -> None:
        result = ToolResult.partial(data={}, message="m", batch_id="b1")
        assert result.metadata["batch_id"] == "b1"


# =============================================================================
# ToolContext
# =============================================================================


class TestToolContext:
    """Tests for the ToolContext dataclass."""

    def test_require_project_raises_when_none(self) -> None:
        ctx = ToolContext(session_id="s", user_id="u", project_id=None)
        with pytest.raises(ValueError, match="No project selected"):
            ctx.require_project()

    def test_require_project_passes_when_set(self) -> None:
        ctx = ToolContext(session_id="s", user_id="u", project_id="proj-1")
        ctx.require_project()  # should not raise

    def test_require_auth_raises_when_none(self) -> None:
        ctx = ToolContext(session_id="s", user_id="u", access_token=None)
        with pytest.raises(ValueError, match="Not authenticated"):
            ctx.require_auth()

    def test_require_auth_passes_when_set(self) -> None:
        ctx = ToolContext(session_id="s", user_id="u", access_token="tok")
        ctx.require_auth()  # should not raise

    def test_default_optional_fields(self) -> None:
        ctx = ToolContext(session_id="s", user_id="u")
        assert ctx.organization_url is None
        assert ctx.project_id is None
        assert ctx.project_name is None
        assert ctx.access_token is None
        assert ctx.conversation_id is None


# =============================================================================
# BaseTool.validate_params
# =============================================================================


class TestBaseToolValidateParams:
    """Tests for BaseTool.validate_params via _DummyTool."""

    def test_missing_required_param_raises(self) -> None:
        tool = _DummyTool()
        with pytest.raises(ValueError, match="Missing required parameter: required_p"):
            tool.validate_params()

    def test_optional_param_gets_default(self) -> None:
        tool = _DummyTool()
        params = tool.validate_params(required_p="hello")
        assert params["optional_p"] == "default_val"

    def test_explicit_optional_param_overrides_default(self) -> None:
        tool = _DummyTool()
        params = tool.validate_params(required_p="hello", optional_p="custom")
        assert params["optional_p"] == "custom"

    def test_extra_params_are_ignored(self) -> None:
        tool = _DummyTool()
        params = tool.validate_params(required_p="val", unexpected_p="ignored")
        assert "unexpected_p" not in params

    def test_all_params_provided(self) -> None:
        tool = _DummyTool()
        params = tool.validate_params(required_p="a", optional_p="b")
        assert params == {"required_p": "a", "optional_p": "b"}

    def test_definition_name_accessible(self) -> None:
        tool = _DummyTool()
        assert tool.definition.name == "dummy"

    async def test_execute_returns_success(self) -> None:
        tool = _DummyTool()
        ctx = ToolContext(session_id="s", user_id="u")
        result = await tool.execute(ctx)
        assert result.status == ToolStatus.SUCCESS
