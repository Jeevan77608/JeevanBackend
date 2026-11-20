"""Tests for src/mcp_tools/registry.py -- tool auto-discovery and registration."""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.mcp_tools.base import BaseTool, ToolContext, ToolDefinition, ToolResult


# =============================================================================
# Concrete dummy tool used across registry tests
# =============================================================================


class _DummyTool(BaseTool):
    """Minimal tool for testing registry mechanics."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(name="dummy_registry", description="test")

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        return ToolResult.success()


# =============================================================================
# register_tools decorator
# =============================================================================


class TestRegisterTools:
    """Tests for the register_tools decorator."""

    def test_register_tools_adds_factory_to_list(self) -> None:
        """Decorating a factory function appends it to _tool_factories."""
        with patch("src.mcp_tools.registry._tool_factories", new=[]) as factories:
            from src.mcp_tools.registry import register_tools

            @register_tools
            def my_factory(session_manager: Any = None) -> list[BaseTool]:
                return [_DummyTool()]

            assert my_factory in factories
            assert len(factories) == 1

    def test_register_tools_returns_original_function(self) -> None:
        """The decorator returns the original callable unchanged."""
        with patch("src.mcp_tools.registry._tool_factories", new=[]):
            from src.mcp_tools.registry import register_tools

            def my_factory(session_manager: Any = None) -> list[BaseTool]:
                return []

            decorated = register_tools(my_factory)
            assert decorated is my_factory

    def test_register_tools_multiple_factories(self) -> None:
        """Multiple decorated factories all end up in the list."""
        with patch("src.mcp_tools.registry._tool_factories", new=[]) as factories:
            from src.mcp_tools.registry import register_tools

            @register_tools
            def factory_a(sm: Any = None) -> list[BaseTool]:
                return []

            @register_tools
            def factory_b(sm: Any = None) -> list[BaseTool]:
                return []

            assert len(factories) == 2
            assert factory_a in factories
            assert factory_b in factories


# =============================================================================
# register_stateless_tools
# =============================================================================


class TestRegisterStatelessTools:
    """Tests for the register_stateless_tools function."""

    def test_register_stateless_tools_adds_tools(self) -> None:
        """Tools are appended to _stateless_tools."""
        with patch("src.mcp_tools.registry._stateless_tools", new=[]) as stateless:
            from src.mcp_tools.registry import register_stateless_tools

            tools = [_DummyTool(), _DummyTool()]
            result = register_stateless_tools(tools)

            assert len(stateless) == 2
            assert result is tools

    def test_register_stateless_tools_returns_same_list(self) -> None:
        """The function returns the exact list passed in."""
        with patch("src.mcp_tools.registry._stateless_tools", new=[]):
            from src.mcp_tools.registry import register_stateless_tools

            tools = [_DummyTool()]
            assert register_stateless_tools(tools) is tools

    def test_register_stateless_tools_accumulates(self) -> None:
        """Calling register_stateless_tools twice extends the list."""
        with patch("src.mcp_tools.registry._stateless_tools", new=[]) as stateless:
            from src.mcp_tools.registry import register_stateless_tools

            register_stateless_tools([_DummyTool()])
            register_stateless_tools([_DummyTool(), _DummyTool()])

            assert len(stateless) == 3


# =============================================================================
# discover_tools
# =============================================================================


class TestDiscoverTools:
    """Tests for the discover_tools function."""

    @patch("src.mcp_tools.registry.importlib.import_module")
    @patch("src.mcp_tools.registry.pkgutil.iter_modules")
    def test_discover_tools_imports_tool_modules(
        self,
        mock_iter: MagicMock,
        mock_import: MagicMock,
    ) -> None:
        """discover_tools imports every module except __init__, base, and registry."""
        mock_iter.return_value = [
            (None, "__init__", False),
            (None, "base", False),
            (None, "registry", False),
            (None, "core", False),
            (None, "context", False),
            (None, "generation", False),
        ]

        from src.mcp_tools.registry import discover_tools

        discover_tools()

        # Only core, context, generation should be imported
        assert mock_import.call_count == 3
        mock_import.assert_any_call("src.mcp_tools.core")
        mock_import.assert_any_call("src.mcp_tools.context")
        mock_import.assert_any_call("src.mcp_tools.generation")

    @patch("src.mcp_tools.registry.importlib.import_module")
    @patch("src.mcp_tools.registry.pkgutil.iter_modules")
    def test_discover_tools_skips_dunder_base_registry(
        self,
        mock_iter: MagicMock,
        mock_import: MagicMock,
    ) -> None:
        """__init__, base, and registry modules are explicitly skipped."""
        mock_iter.return_value = [
            (None, "__init__", False),
            (None, "base", False),
            (None, "registry", False),
        ]

        from src.mcp_tools.registry import discover_tools

        discover_tools()

        mock_import.assert_not_called()

    @patch("src.mcp_tools.registry.importlib.import_module", side_effect=ImportError("oops"))
    @patch("src.mcp_tools.registry.pkgutil.iter_modules")
    def test_discover_tools_logs_import_errors(
        self,
        mock_iter: MagicMock,
        mock_import: MagicMock,
    ) -> None:
        """ImportError is caught and logged, not raised."""
        mock_iter.return_value = [(None, "broken_module", False)]

        from src.mcp_tools.registry import discover_tools

        # Should not raise
        discover_tools()

        mock_import.assert_called_once_with("src.mcp_tools.broken_module")

    @patch("src.mcp_tools.registry.importlib.import_module")
    @patch("src.mcp_tools.registry.pkgutil.iter_modules")
    def test_discover_tools_with_empty_package(
        self,
        mock_iter: MagicMock,
        mock_import: MagicMock,
    ) -> None:
        """When there are no modules, nothing is imported."""
        mock_iter.return_value = []

        from src.mcp_tools.registry import discover_tools

        discover_tools()

        mock_import.assert_not_called()


# =============================================================================
# create_all_tools
# =============================================================================


class TestCreateAllTools:
    """Tests for the create_all_tools function."""

    def test_create_all_tools_combines_stateless_and_factory(self) -> None:
        """create_all_tools returns stateless tools plus factory-produced tools."""
        stateless_tool = _DummyTool()
        factory_tool = _DummyTool()

        def test_factory(session_manager: Any) -> list[BaseTool]:
            return [factory_tool]

        with (
            patch("src.mcp_tools.registry._stateless_tools", new=[stateless_tool]),
            patch("src.mcp_tools.registry._tool_factories", new=[test_factory]),
        ):
            from src.mcp_tools.registry import create_all_tools

            mock_sm = MagicMock()
            tools = create_all_tools(mock_sm)

            assert stateless_tool in tools
            assert factory_tool in tools
            assert len(tools) == 2

    def test_create_all_tools_passes_session_manager_to_factories(self) -> None:
        """Each factory receives the session_manager argument."""
        mock_sm = MagicMock()
        factory = MagicMock(return_value=[])

        with (
            patch("src.mcp_tools.registry._stateless_tools", new=[]),
            patch("src.mcp_tools.registry._tool_factories", new=[factory]),
        ):
            from src.mcp_tools.registry import create_all_tools

            create_all_tools(mock_sm)

            factory.assert_called_once_with(mock_sm)

    def test_create_all_tools_no_factories_returns_stateless_only(self) -> None:
        """When no factories are registered, only stateless tools are returned."""
        tool = _DummyTool()

        with (
            patch("src.mcp_tools.registry._stateless_tools", new=[tool]),
            patch("src.mcp_tools.registry._tool_factories", new=[]),
        ):
            from src.mcp_tools.registry import create_all_tools

            tools = create_all_tools(MagicMock())
            assert tools == [tool]

    def test_create_all_tools_empty_registries_returns_empty(self) -> None:
        """With nothing registered, an empty list is returned."""
        with (
            patch("src.mcp_tools.registry._stateless_tools", new=[]),
            patch("src.mcp_tools.registry._tool_factories", new=[]),
        ):
            from src.mcp_tools.registry import create_all_tools

            tools = create_all_tools(MagicMock())
            assert tools == []

    def test_create_all_tools_multiple_factories(self) -> None:
        """Tools from multiple factories are all included."""
        tool_a = _DummyTool()
        tool_b = _DummyTool()

        def factory_a(sm: Any) -> list[BaseTool]:
            return [tool_a]

        def factory_b(sm: Any) -> list[BaseTool]:
            return [tool_b]

        with (
            patch("src.mcp_tools.registry._stateless_tools", new=[]),
            patch("src.mcp_tools.registry._tool_factories", new=[factory_a, factory_b]),
        ):
            from src.mcp_tools.registry import create_all_tools

            tools = create_all_tools(MagicMock())
            assert len(tools) == 2
            assert tool_a in tools
            assert tool_b in tools

    def test_create_all_tools_does_not_mutate_stateless_list(self) -> None:
        """The original _stateless_tools list is not modified."""
        original = [_DummyTool()]

        def factory(sm: Any) -> list[BaseTool]:
            return [_DummyTool()]

        with (
            patch("src.mcp_tools.registry._stateless_tools", new=original),
            patch("src.mcp_tools.registry._tool_factories", new=[factory]),
        ):
            from src.mcp_tools.registry import create_all_tools

            create_all_tools(MagicMock())
            # Original list should still have exactly 1 item
            assert len(original) == 1
