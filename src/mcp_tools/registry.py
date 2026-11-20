"""Tool auto-discovery registry for MCP tools."""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.session.manager import SessionManager

from src.mcp_tools.base import BaseTool

logger = logging.getLogger(__name__)

# Module-level registries populated by decorators at import time
_tool_factories: list[Callable[..., list[BaseTool]]] = []
_stateless_tools: list[BaseTool] = []


def register_tools(factory: Callable[..., list[BaseTool]]) -> Callable[..., list[BaseTool]]:
    """Decorator for tool factory functions that need a session_manager."""
    _tool_factories.append(factory)
    return factory


def register_stateless_tools(tools: list[BaseTool]) -> list[BaseTool]:
    """Register tools that do not need session_manager."""
    _stateless_tools.extend(tools)
    return tools


def discover_tools() -> None:
    """
    Import all modules in the mcp_tools package to trigger registration.
    Called once at startup.
    """
    import src.mcp_tools as pkg

    for _importer, modname, _ispkg in pkgutil.iter_modules(pkg.__path__):
        if modname in ("__init__", "base", "registry"):
            continue
        try:
            importlib.import_module(f"src.mcp_tools.{modname}")
        except ImportError as e:
            logger.warning("Could not import tool module %s: %s", modname, e)


def create_all_tools(session_manager: SessionManager) -> list[BaseTool]:
    """Create all registered tools with the given session_manager."""
    tools: list[BaseTool] = list(_stateless_tools)
    for factory in _tool_factories:
        tools.extend(factory(session_manager))
    return tools
