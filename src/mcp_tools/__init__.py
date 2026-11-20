"""MCP tools for Jira and Azure DevOps integrations.

This package provides Model Context Protocol (MCP) tools for AI-powered
interaction with connected work-management platforms. The current assistant
workflow uses Jira; Azure DevOps tools remain available for compatibility.

Tool Categories:
- Jira: Jira projects, issues, comments, and draft creation
- Core: User profile, organizations, projects
- Board & Wiki: Board structure, wiki pages
- Context: Work item retrieval for AI context
- Generation: AI-powered work item generation
- Edit & Review: Work item editing and validation
- Push: Push work items to Azure DevOps
"""

from src.mcp_tools.base import (
    BaseTool,
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    ToolStatus,
)
from src.mcp_tools.registry import create_all_tools, discover_tools

# Auto-discover all tool modules (triggers their register_tools decorators)
discover_tools()

__all__ = [
    "BaseTool",
    "ToolContext",
    "ToolDefinition",
    "ToolParameter",
    "ToolResult",
    "ToolStatus",
    "create_all_tools",
]
