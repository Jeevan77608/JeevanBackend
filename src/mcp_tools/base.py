"""
Base classes and utilities for MCP Tools.

This module provides the foundation for all MCP (Model Context Protocol) tools
in the Azure DevOps MCP Server. It defines the core abstractions:

- ToolStatus: Enum for tool execution status (success, error, partial)
- ToolResult: Standardized response from tool execution
- ToolContext: Request context with user, session, and project info
- ToolDefinition: Tool metadata for discovery and documentation
- ToolParameter: Parameter schema for tool inputs
- BaseTool: Abstract base class for implementing tools

Design Patterns:
- Factory Method: ToolResult classmethods for creating results
- Template Method: BaseTool.execute() with validate_params() hook
- Strategy: Different tool implementations with same interface

Usage:
    class MyTool(BaseTool):
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(
                name="my_tool",
                description="Does something useful",
                parameters=[
                    ToolParameter(name="input", type="string", description="Input value"),
                ],
            )

        async def execute(self, context: ToolContext, **kwargs) -> ToolResult:
            params = self.validate_params(**kwargs)
            # Do work...
            return ToolResult.success(data=result)

References:
    - Model Context Protocol: https://modelcontextprotocol.io/
    - Azure DevOps REST API: https://learn.microsoft.com/en-us/rest/api/azure/devops/
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Result Types
# =============================================================================


class ToolStatus(str, Enum):
    """
    Execution status of an MCP tool.

    Values:
        SUCCESS: Tool completed successfully with valid data
        ERROR: Tool failed with an error message
        PARTIAL: Some operations succeeded, some failed (batch operations)
    """

    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


@dataclass
class ToolResult:
    """
    Standardized result from tool execution.

    All MCP tools return a ToolResult to provide consistent response structure
    for AI models to interpret. The result includes:

    - status: Whether the tool succeeded, failed, or partially succeeded
    - data: The actual result data (type varies by tool)
    - message: Human-readable success message
    - error: Error description if status is ERROR
    - metadata: Additional context (e.g., pagination, update flags)

    Use the factory classmethods (success, failure, partial) instead of
    constructing directly for cleaner code.

    Example:
        # Success with data
        return ToolResult.success(
            data={"items": work_items},
            message=f"Found {len(work_items)} items",
        )

        # Error
        return ToolResult.failure("Work item not found")

        # Partial success (batch operations)
        return ToolResult.partial(
            data={"succeeded": 3, "failed": 1},
            message="3 of 4 items pushed successfully",
        )
    """

    status: ToolStatus
    data: Any = None
    message: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(
        cls,
        data: Any = None,
        message: str | None = None,
        **metadata: Any,
    ) -> ToolResult:
        """
        Create a success result.

        Args:
            data: Result data to return to the AI model
            message: Human-readable success message
            **metadata: Additional context (e.g., update_context=True)

        Returns:
            ToolResult with SUCCESS status
        """
        return cls(
            status=ToolStatus.SUCCESS,
            data=data,
            message=message,
            metadata=metadata,
        )

    @classmethod
    def failure(
        cls,
        error: str,
        data: Any = None,
        **metadata: Any,
    ) -> ToolResult:
        """
        Create an error result.

        Args:
            error: Error message describing what went wrong
            data: Optional partial data or error details
            **metadata: Additional context

        Returns:
            ToolResult with ERROR status
        """
        return cls(
            status=ToolStatus.ERROR,
            error=error,
            data=data,
            metadata=metadata,
        )

    @classmethod
    def partial(
        cls,
        data: Any,
        message: str,
        **metadata: Any,
    ) -> ToolResult:
        """
        Create a partial success result.

        Used when some operations in a batch succeeded while others failed.

        Args:
            data: Results including both successes and failures
            message: Description of what succeeded/failed
            **metadata: Additional context

        Returns:
            ToolResult with PARTIAL status
        """
        return cls(
            status=ToolStatus.PARTIAL,
            data=data,
            message=message,
            metadata=metadata,
        )


# =============================================================================
# Tool Context
# =============================================================================


@dataclass
class ToolContext:
    """
    Execution context passed to every tool.

    Contains all the information a tool needs to execute:
    - Session and user identification
    - Selected organization and project
    - Authentication token for API calls
    - Conversation tracking for multi-turn interactions

    The context is built from the authenticated session and passed
    to each tool's execute() method.

    Attributes:
        session_id: Unique session identifier
        user_id: Azure AD user ID
        organization_url: Azure DevOps organization URL (e.g., https://dev.azure.com/org)
        project_id: Selected project ID or name
        project_name: Human-readable project name
        access_token: OAuth/PAT token for Azure DevOps API
        conversation_id: Optional conversation ID for context tracking
    """

    session_id: str
    user_id: str
    organization_url: str | None = None
    project_id: str | None = None
    project_name: str | None = None
    access_token: str | None = None
    conversation_id: str | None = None

    def require_project(self) -> None:
        """
        Validate that a project is selected.

        Many tools require an active project. Call this at the start
        of execute() for project-scoped operations.

        Raises:
            ValueError: If no project is selected
        """
        if not self.project_id:
            raise ValueError("No project selected. Use select_project first.")

    def require_auth(self) -> None:
        """
        Validate that the user is authenticated.

        All tools that make API calls should call this.

        Raises:
            ValueError: If no access token is available
        """
        if not self.access_token:
            raise ValueError("Not authenticated. Please login first.")


# =============================================================================
# Tool Definition (OpenAPI-style schema)
# =============================================================================


class ToolParameter(BaseModel):
    """
    Definition of a tool input parameter.

    Used to generate tool schemas for AI model consumption and
    to validate incoming parameters.

    Attributes:
        name: Parameter name (should be snake_case)
        type: JSON Schema type (string, number, boolean, array, object)
        description: Human-readable description for AI
        required: Whether the parameter must be provided
        default: Default value when not required
        enum: List of allowed values for constrained inputs
    """

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (string, number, boolean, array, object)")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")
    default: Any = Field(default=None, description="Default value if not required")
    enum: list[str] | None = Field(default=None, description="Allowed values for enum types")


class ToolDefinition(BaseModel):
    """
    Complete definition of an MCP tool.

    This metadata is exposed through the /tools endpoint and used by
    AI models to understand what tools are available and how to use them.

    Attributes:
        name: Unique tool name in snake_case (e.g., "get_work_item")
        description: Detailed description for AI to understand when to use the tool
        parameters: List of input parameters
        returns: Description of what the tool returns
        examples: Natural language examples of when to use the tool
        category: Tool category for grouping (core, context, generation, etc.)
    """

    name: str = Field(..., description="Tool name (snake_case)")
    description: str = Field(..., description="Tool description for AI")
    parameters: list[ToolParameter] = Field(default_factory=list)
    returns: str = Field(default="ToolResult", description="Return type description")
    examples: list[str] = Field(default_factory=list, description="Usage examples")
    category: str = Field(default="general", description="Tool category")


# =============================================================================
# Base Tool Class (Template Method Pattern)
# =============================================================================


class BaseTool(ABC):
    """
    Abstract base class for all MCP tools.

    Implements the Template Method pattern where:
    - definition property provides tool metadata
    - execute() is the main entry point
    - validate_params() provides common parameter validation

    Subclasses must implement:
    - definition: Property returning ToolDefinition
    - execute(): Async method performing the actual work

    Example:
        class MyTool(BaseTool):
            @property
            def definition(self) -> ToolDefinition:
                return ToolDefinition(
                    name="my_tool",
                    description="Does something useful",
                    parameters=[...],
                    category="custom",
                )

            async def execute(self, context: ToolContext, **kwargs) -> ToolResult:
                # Validate required context
                context.require_auth()
                context.require_project()

                # Validate and extract parameters
                params = self.validate_params(**kwargs)
                my_param = params["my_param"]

                # Do the actual work
                try:
                    result = await do_something(my_param)
                    return ToolResult.success(data=result)
                except Exception as e:
                    return ToolResult.failure(str(e))
    """

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """
        Get the tool definition.

        Returns:
            ToolDefinition with name, description, parameters, etc.
        """
        pass

    @abstractmethod
    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Execute the tool.

        Args:
            context: Execution context with session, user, project info
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult with status, data, and optional message/error
        """
        pass

    def validate_params(self, **kwargs: Any) -> dict[str, Any]:
        """
        Validate and normalize input parameters.

        Checks that all required parameters are present and applies
        default values for optional parameters.

        Args:
            **kwargs: Raw parameters from the request

        Returns:
            Dict of validated parameters with defaults applied

        Raises:
            ValueError: If a required parameter is missing
        """
        validated = {}
        for param in self.definition.parameters:
            value = kwargs.get(param.name)
            if value is None and param.required:
                raise ValueError(f"Missing required parameter: {param.name}")
            if value is None:
                value = param.default
            validated[param.name] = value
        return validated

