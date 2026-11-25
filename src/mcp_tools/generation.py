"""
AI Generation MCP Tools for Azure DevOps.

This module provides AI-powered work item generation using Azure OpenAI.
It is the core value proposition of the MCP server, enabling AI assistants
to generate high-quality Azure DevOps work items from natural language.

Tools:
    - generate_user_stories: Generate user stories from requirements text
    - generate_tasks: Generate tasks from a single user story or feature
    - generate_tasks_batch: Generate tasks for ALL parent items in one call
    - generate_bug_report: Generate structured bug reports from descriptions
    - refine_item: Refine a draft work item based on feedback

Key Features:
    - Context-aware generation using existing work items as reference
    - Structured output following Azure DevOps work item schemas
    - Draft management for review before pushing to DevOps
    - Support for acceptance criteria, story points, and priorities

Architecture:
    1. User provides requirements/description + optional source work item IDs
    2. Tool fetches source work items for context (if provided)
    3. Azure OpenAI generates structured work item(s)
    4. Generated items are stored as drafts in the session
    5. User can review, refine, and push drafts to Azure DevOps

Example Flow:
    1. generate_user_stories(requirements="Build login page", source_work_item_ids=[1001])
    2. list_drafts() -> Shows generated user stories
    3. refine_item(draft_id="abc", feedback="Add OAuth support")
    4. push_to_devops(area_path="MyProject\\Team")

Dependencies:
    - Azure OpenAI (GPT-4 recommended)
    - Azure DevOps REST API (for context fetching)
    - Session Manager (for draft storage)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.clients.devops import AzureDevOpsClient, AzureDevOpsError
from src.clients.openai import (
    MAX_FILE_CONTENT_CHARS,
    MAX_USER_CONTEXT_CHARS,
    MAX_WORK_ITEM_CONTEXT_CHARS,
    AzureOpenAIClient,
    ContextLengthError,
    GenerationError,
    OpenAIError,
    truncate_context,
)
from src.config.settings import get_settings
from src.mcp_tools.base import (
    BaseTool,
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from src.mcp_tools.conversations import sync_drafts_to_conversation
from src.mcp_tools.registry import register_tools
from src.models.work_items import DraftCollection, DraftWorkItem
from src.session.manager import SessionManager

logger = logging.getLogger(__name__)


async def add_draft_to_session(
    session_manager: SessionManager,
    session_id: str,
    project_id: str,
    draft_item: DraftWorkItem,
) -> None:
    """Add a draft item to session, creating collection if needed."""
    drafts = await session_manager.get_drafts(session_id)
    if not drafts:
        # Create new draft collection
        drafts = DraftCollection(
            session_id=session_id,
            project_id=project_id,
            items=[draft_item],
        )
        await session_manager.save_drafts(session_id, drafts)
    else:
        # Add to existing collection
        drafts.items.append(draft_item)
        await session_manager.save_drafts(session_id, drafts)


async def fetch_work_item_context(
    context: ToolContext,
    work_item_ids: list[int],
) -> str:
    """
    Fetch work items and convert them to context strings for AI generation.

    Args:
        context: Tool context with auth and project info
        work_item_ids: List of work item IDs to fetch

    Returns:
        Combined context string from all work items
    """
    if not work_item_ids:
        return ""

    settings = get_settings()
    context_parts = []

    # Use PAT auth when OAuth is disabled (local development)
    use_pat = not settings.auth_enabled

    try:
        async with AzureDevOpsClient(
            settings=settings.azure_devops,
            access_token=context.access_token,
            organization_url=context.organization_url,
            use_pat=use_pat,
        ) as client:
            work_items = await client.get_work_items_batch(
                work_item_ids=work_item_ids,
                project=context.project_name,
            )

            for wi in work_items:
                context_parts.append(wi.to_context_string())

    except AzureDevOpsError as e:
        logger.warning(f"Failed to fetch work item context: {e}")

    return "\n\n---\n\n".join(context_parts)


# =============================================================================
# Generate User Stories Tool
# =============================================================================


class GenerateUserStoriesTool(BaseTool):
    """Generate user stories from requirements."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="generate_user_stories",
            description=(
                "Generate user stories or Epics from natural language requirements using AI. "
                "Follow the requested work item type and count; use existing work items as context when provided."
            ),
            parameters=[
                ToolParameter(
                    name="requirements",
                    type="string",
                    description="Natural language description of requirements",
                    required=True,
                ),
                ToolParameter(
                    name="additional_context",
                    type="string",
                    description="Additional context (project info, constraints, existing features, file content)",
                    required=False,
                ),
                ToolParameter(
                    name="file_id",
                    type="string",
                    description="ID of uploaded file to use as context (filename)",
                    required=False,
                ),
                ToolParameter(
                    name="source_work_item_ids",
                    type="array",
                    description=(
                        "IDs of existing work items (Epics, Stories) to use as context. "
                        "The AI will consider these when generating new stories."
                    ),
                    required=False,
                ),
                ToolParameter(
                    name="count",
                    type="integer",
                    description="Exact number of work items to generate (1-10), when specified by the user.",
                    required=False,
                ),
            ],
            returns="Generated user stories or Epics as draft work items",
            category="generation",
            examples=[
                "Generate user stories for a login feature",
                "Create stories for shopping cart functionality",
                "Generate stories based on Epic #100",
            ],
        )

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute generate_user_stories."""
        context.require_project()

        params = self.validate_params(**kwargs)
        requirements = params["requirements"]
        additional_context = params.get("additional_context") or ""
        source_work_item_ids = params.get("source_work_item_ids") or []
        file_id = params.get("file_id")
        count = params.get("count")
        if count is not None and (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 1 <= count <= 10
        ):
            return ToolResult.failure("Count must be an integer between 1 and 10.")

        # Build structured context from all sources (with size limits)
        context_sections: list[str] = []

        # 1. File content (highest detail — primary source when present)
        if file_id:
            file_content = await self._session_manager.get_file_content(
                session_id=context.session_id,
                file_id=file_id,
            )
            if file_content:
                file_content = truncate_context(file_content, MAX_FILE_CONTENT_CHARS, f"file '{file_id}'")
                context_sections.append(
                    f"[SOURCE: Uploaded File — {file_id}]\n{file_content}\n[END SOURCE]"
                )

        # 2. Existing work items (reference material)
        if source_work_item_ids:
            work_item_context = await fetch_work_item_context(
                context=context,
                work_item_ids=[int(id) for id in source_work_item_ids],
            )
            if work_item_context:
                work_item_context = truncate_context(work_item_context, MAX_WORK_ITEM_CONTEXT_CHARS, "work item context")
                context_sections.append(
                    f"[SOURCE: Existing Work Items]\n{work_item_context}\n[END SOURCE]"
                )

        # 3. Additional context from user (instructions, wiki content, typed requirements)
        if additional_context:
            additional_context = truncate_context(additional_context, MAX_USER_CONTEXT_CHARS, "user-provided context")
            context_sections.append(
                f"[SOURCE: User-Provided Context]\n{additional_context}\n[END SOURCE]"
            )

        # Combine with clear separation
        if context_sections:
            additional_context = (
                "The following context sources are provided. "
                "Generate work items that are grounded in these sources. "
                "Cross-reference multiple sources for accuracy.\n\n"
                + "\n\n".join(context_sections)
            )
        else:
            additional_context = ""

        settings = get_settings()

        # Fetch board context for dynamic prompts
        board_context = None
        if context.project_name:
            board_context = await self._session_manager.get_board_context(
                session_id=context.session_id,
                project_id=context.project_name,
            )

        try:
            async with AzureOpenAIClient(settings.azure_openai) as client:
                result = await client.generate_user_stories(
                    requirements=requirements,
                    context=additional_context,
                    board_context=board_context,
                    count=count,
                )

            # Store drafts in session
            for item in result.work_items:
                await add_draft_to_session(
                    session_manager=self._session_manager,
                    session_id=context.session_id,
                    project_id=context.project_name or "unknown",
                    draft_item=item,
                )

            await sync_drafts_to_conversation(context, self._session_manager)
            return ToolResult.success(
                data={
                    "work_items": [
                        {
                            "draft_id": item.draft_id,
                            "type": item.work_item_type,
                            "title": item.title,
                            "description": item.description,
                            "acceptance_criteria": item.acceptance_criteria,
                            "story_points": item.story_points,
                            "priority": item.priority,
                        }
                        for item in result.work_items
                    ],
                    "usage": {
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "model": result.model,
                    },
                    "context_source_ids": source_work_item_ids,
                },
                message=f"Generated {len(result.work_items)} user story(ies)",
            )

        except ContextLengthError as e:
            logger.warning(f"Context too large: {e}")
            return ToolResult.failure(str(e))
        except GenerationError as e:
            logger.error(f"Generation failed: {e}")
            return ToolResult.failure(f"Failed to generate user stories: {e}")
        except OpenAIError as e:
            logger.error(f"OpenAI error: {e}")
            return ToolResult.failure(f"AI service error: {e}")


# =============================================================================
# Generate Tasks Tool
# =============================================================================


class GenerateTasksTool(BaseTool):
    """Generate tasks from a user story."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="generate_tasks",
            description=(
                "Generate implementation tasks from a user story using AI. "
                "IMPORTANT: If the user references a user story by number (e.g., 'user story 4', 'first user story') "
                "or by title, you MUST first call list_drafts to get all user stories, find the matching one, "
                "and use its draft_id as the user_story parameter. This ensures tasks are properly linked to the parent user story. "
                "The user_story parameter can be: (1) a draft_id (preferred - use this when available), "
                "(2) user story text/title, or (3) a reference like 'user story 4' (requires calling list_drafts first)."
            ),
            parameters=[
                ToolParameter(
                    name="user_story",
                    type="string",
                    description=(
                        "User story draft ID (preferred), user story text/title, or reference. "
                        "If user references by number/title, first call list_drafts to find the draft_id. "
                        "Example draft_id: 'draft-abc123' or 'abc-123-def-456'. "
                        "When a draft_id is used, tasks will be automatically linked to that user story."
                    ),
                    required=True,
                ),
                ToolParameter(
                    name="additional_context",
                    type="string",
                    description="Additional technical context",
                    required=False,
                ),
                ToolParameter(
                    name="source_work_item_ids",
                    type="array",
                    description=(
                        "IDs of existing work items to use as additional context. "
                        "Useful for referencing related stories or tasks."
                    ),
                    required=False,
                ),
            ],
            returns="Generated tasks as draft work items, linked to parent user story if draft_id was used",
            category="generation",
            examples=[
                "Generate tasks for draft_id 'draft-abc123'",
                "Generate tasks for the first user story (first call list_drafts, then use its draft_id)",
                "Generate tasks for user story 4 (first call list_drafts, find the 4th one, use its draft_id)",
            ],
        )

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute generate_tasks."""
        context.require_auth()
        context.require_project()

        params = self.validate_params(**kwargs)
        user_story = params["user_story"]
        additional_context = params.get("additional_context") or ""
        source_work_item_ids = params.get("source_work_item_ids") or []

        # Fetch board context to get parent types dynamically
        board_context = None
        parent_types_for_tasks = ["User Story"]  # Default
        if context.project_name:
            board_context = await self._session_manager.get_board_context(
                session_id=context.session_id,
                project_id=context.project_name,
            )
            if board_context:
                parent_types_for_tasks = board_context.parent_types_for_tasks

        # Epic and Feature can also be parents for tasks (e.g., Epic -> Task when no User Story level)
        accepted_parent_types = set(parent_types_for_tasks) | {"Epic", "Feature"}

        # Check if user_story is a draft_id and get the parent draft
        parent_draft_id: str | None = None
        user_story_text = user_story

        # First, try to get draft by ID - if it exists and is a valid parent type, use it as parent
        # Check if user_story looks like a draft ID (UUID format or starts with "draft-")
        if user_story and (user_story.startswith("draft-") or len(user_story) > 20):
            try:
                parent_draft = await self._session_manager.get_draft(
                    session_id=context.session_id,
                    draft_id=user_story,
                )
                # Accept parent types from board context + Epic/Feature (for Epic -> Task hierarchy)
                if parent_draft and parent_draft.work_item_type in accepted_parent_types:
                    parent_draft_id = parent_draft.draft_id
                    # Use the draft's title and description as the user story text
                    user_story_text = f"{parent_draft.title}\n{parent_draft.description or ''}"
            except Exception as e:
                # If it's not a draft_id or draft doesn't exist, try to match by title
                logger.debug(f"Could not get draft {user_story}: {e}")
                pass

        # If not found by ID, try to match by title, partial title, or numeric reference
        if not parent_draft_id:
            try:
                draft_collection = await self._session_manager.get_drafts(
                    session_id=context.session_id,
                )
                if draft_collection:
                    # Look for work items that match the user_story text (any valid parent type)
                    # Sort by created_at to ensure consistent ordering for numeric references
                    parent_items = sorted(
                        [d for d in draft_collection.items if d.work_item_type in accepted_parent_types],
                        key=lambda x: x.created_at
                    )

                    # Check for numeric reference (e.g., "user story 4", "4th user story", "story 4")
                    number_match = re.search(r'\d+', user_story.lower())
                    if number_match:
                        try:
                            story_index = int(number_match.group()) - 1  # Convert to 0-based index
                            if 0 <= story_index < len(parent_items):
                                story = parent_items[story_index]
                                parent_draft_id = story.draft_id
                                user_story_text = f"{story.title}\n{story.description or ''}"
                                logger.info(f"Matched parent item by number: {story_index + 1} -> {story.draft_id} ({story.title})")
                        except (ValueError, IndexError):
                            pass

                    # Try exact title match
                    if not parent_draft_id:
                        for story in parent_items:
                            if story.title.lower() == user_story.lower() or story.title.lower().startswith(user_story.lower()):
                                parent_draft_id = story.draft_id
                                user_story_text = f"{story.title}\n{story.description or ''}"
                                break

                    # If still not found, try partial match in title or description
                    if not parent_draft_id:
                        user_story_lower = user_story.lower()
                        for story in parent_items:
                            if (user_story_lower in story.title.lower() or
                                (story.description and user_story_lower in story.description.lower())):
                                parent_draft_id = story.draft_id
                                user_story_text = f"{story.title}\n{story.description or ''}"
                                break
            except Exception as e:
                logger.debug(f"Could not match user story by title: {e}")
                pass

        # Build structured context from all sources (with size limits)
        context_sections: list[str] = []

        if source_work_item_ids:
            work_item_context = await fetch_work_item_context(
                context=context,
                work_item_ids=[int(id) for id in source_work_item_ids],
            )
            if work_item_context:
                work_item_context = truncate_context(work_item_context, MAX_WORK_ITEM_CONTEXT_CHARS, "work item context")
                context_sections.append(
                    f"[SOURCE: Related Work Items]\n{work_item_context}\n[END SOURCE]"
                )

        if additional_context:
            additional_context = truncate_context(additional_context, MAX_USER_CONTEXT_CHARS, "additional context")
            context_sections.append(
                f"[SOURCE: Additional Context]\n{additional_context}\n[END SOURCE]"
            )

        if context_sections:
            additional_context = "\n\n".join(context_sections)
        else:
            additional_context = ""

        settings = get_settings()

        try:
            # Use parent type for prompt when we have a parent draft (e.g. Epic vs User Story)
            parent_type_for_prompt = None
            if parent_draft_id:
                try:
                    parent_draft = await self._session_manager.get_draft(
                        session_id=context.session_id,
                        draft_id=parent_draft_id,
                    )
                    if parent_draft:
                        parent_type_for_prompt = parent_draft.work_item_type
                except Exception:
                    pass

            async with AzureOpenAIClient(settings.azure_openai) as client:
                result = await client.generate_tasks(
                    user_story=user_story_text,
                    context=additional_context,
                    board_context=board_context,
                    parent_type=parent_type_for_prompt,
                )

            # Set parent_draft_id on generated tasks if we found a parent
            if parent_draft_id:
                for item in result.work_items:
                    item.parent_draft_id = parent_draft_id

            # Store drafts in session
            for item in result.work_items:
                await add_draft_to_session(
                    session_manager=self._session_manager,
                    session_id=context.session_id,
                    project_id=context.project_name or "unknown",
                    draft_item=item,
                )

            await sync_drafts_to_conversation(context, self._session_manager)
            return ToolResult.success(
                data={
                    "work_items": [
                        {
                            "draft_id": item.draft_id,
                            "type": item.work_item_type,
                            "title": item.title,
                            "description": item.description,
                            "parent_draft_id": item.parent_draft_id,
                        }
                        for item in result.work_items
                    ],
                    "usage": {
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "model": result.model,
                    },
                    "context_source_ids": source_work_item_ids,
                },
                message=f"Generated {len(result.work_items)} task(s)",
            )

        except ContextLengthError as e:
            logger.warning(f"Context too large: {e}")
            return ToolResult.failure(str(e))
        except GenerationError as e:
            logger.error(f"Generation failed: {e}")
            return ToolResult.failure(f"Failed to generate tasks: {e}")
        except OpenAIError as e:
            logger.error(f"OpenAI error: {e}")
            return ToolResult.failure(f"AI service error: {e}")


# =============================================================================
# Generate Bug Report Tool
# =============================================================================


class GenerateBugReportTool(BaseTool):
    """Generate bug report from issue description."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="generate_bug_report",
            description=(
                "Generate a structured bug report from an issue description using AI. "
                "Can reference existing work items for context."
            ),
            parameters=[
                ToolParameter(
                    name="issue_description",
                    type="string",
                    description="Description of the issue or bug",
                    required=True,
                ),
                ToolParameter(
                    name="additional_context",
                    type="string",
                    description="Additional context (environment, steps to reproduce)",
                    required=False,
                ),
                ToolParameter(
                    name="source_work_item_ids",
                    type="array",
                    description=(
                        "IDs of related work items (e.g., the User Story where the bug was found)"
                    ),
                    required=False,
                ),
            ],
            returns="Generated bug report as draft work item",
            category="generation",
            examples=[
                "Create a bug report for the login timeout issue",
                "Generate bug for the payment processing error",
                "Create bug report related to User Story #150",
            ],
        )

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute generate_bug_report."""
        context.require_auth()
        context.require_project()

        params = self.validate_params(**kwargs)
        issue_description = params["issue_description"]
        additional_context = params.get("additional_context") or ""
        source_work_item_ids = params.get("source_work_item_ids") or []

        # Build structured context from all sources (with size limits)
        context_sections: list[str] = []

        if source_work_item_ids:
            work_item_context = await fetch_work_item_context(
                context=context,
                work_item_ids=[int(id) for id in source_work_item_ids],
            )
            if work_item_context:
                work_item_context = truncate_context(work_item_context, MAX_WORK_ITEM_CONTEXT_CHARS, "work item context")
                context_sections.append(
                    f"[SOURCE: Related Work Items]\n{work_item_context}\n[END SOURCE]"
                )

        if additional_context:
            additional_context = truncate_context(additional_context, MAX_USER_CONTEXT_CHARS, "additional context")
            context_sections.append(
                f"[SOURCE: Additional Context]\n{additional_context}\n[END SOURCE]"
            )

        if context_sections:
            additional_context = "\n\n".join(context_sections)
        else:
            additional_context = ""

        settings = get_settings()

        # Fetch board context for dynamic prompts
        board_context = None
        if context.project_name:
            board_context = await self._session_manager.get_board_context(
                session_id=context.session_id,
                project_id=context.project_name,
            )

        try:
            async with AzureOpenAIClient(settings.azure_openai) as client:
                result = await client.generate_bug_report(
                    issue_description=issue_description,
                    context=additional_context,
                    board_context=board_context,
                )

            # Store drafts in session
            for item in result.work_items:
                await add_draft_to_session(
                    session_manager=self._session_manager,
                    session_id=context.session_id,
                    project_id=context.project_name or "unknown",
                    draft_item=item,
                )

            await sync_drafts_to_conversation(context, self._session_manager)
            return ToolResult.success(
                data={
                    "work_items": [
                        {
                            "draft_id": item.draft_id,
                            "type": item.work_item_type,
                            "title": item.title,
                            "description": item.description,
                            "priority": item.priority,
                        }
                        for item in result.work_items
                    ],
                    "usage": {
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "model": result.model,
                    },
                    "context_source_ids": source_work_item_ids,
                },
                message=f"Generated {len(result.work_items)} bug report(s)",
            )

        except ContextLengthError as e:
            logger.warning(f"Context too large: {e}")
            return ToolResult.failure(str(e))
        except GenerationError as e:
            logger.error(f"Generation failed: {e}")
            return ToolResult.failure(f"Failed to generate bug report: {e}")
        except OpenAIError as e:
            logger.error(f"OpenAI error: {e}")
            return ToolResult.failure(f"AI service error: {e}")


class GenerateTestCasesTool(BaseTool):
    """Generate independently reviewable QA test case drafts."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="generate_test_cases",
            description="Create deduplicated, executable positive, negative, boundary, and permission test cases from a requirement or user story. Each case is saved as a reviewable Jira draft.",
            parameters=[
                ToolParameter(name="requirement", type="string", description="User story or requirement to test", required=True),
                ToolParameter(name="additional_context", type="string", description="Environment, constraints, or test data context", required=False),
                ToolParameter(name="count", type="integer", description="Requested number of cases (1-20)", required=False),
            ],
            returns="Test case drafts with preconditions, data, steps, and expected results",
            category="generation",
            examples=["Create test cases for the certificate generation story"],
        )

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        context.require_auth()
        context.require_project()
        params = self.validate_params(**kwargs)
        count = params.get("count")
        if count is not None and (isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 20):
            return ToolResult.failure("Count must be an integer between 1 and 20.")
        try:
            async with AzureOpenAIClient(get_settings().azure_openai) as client:
                result = await client.generate_test_cases(
                    requirement=params["requirement"],
                    context=params.get("additional_context"),
                    count=count,
                )
            for item in result.work_items:
                await add_draft_to_session(self._session_manager, context.session_id, context.project_name or "unknown", item)
            await sync_drafts_to_conversation(context, self._session_manager)
            return ToolResult.success(
                data={"work_items": [{"draft_id": item.draft_id, "type": item.work_item_type, "title": item.title, "description": item.description, "acceptance_criteria": item.acceptance_criteria, "priority": item.priority} for item in result.work_items], "usage": {"model": result.model, "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens}},
                message=f"Generated {len(result.work_items)} test case(s)",
            )
        except (ContextLengthError, GenerationError, OpenAIError) as error:
            logger.error("Test case generation failed: %s", error)
            return ToolResult.failure(f"Failed to generate test cases: {error}")


# =============================================================================
# Refine Item Tool
# =============================================================================


class RefineItemTool(BaseTool):
    """Refine a draft work item based on feedback."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="refine_item",
            description="Refine a draft work item based on user feedback using AI.",
            parameters=[
                ToolParameter(
                    name="draft_id",
                    type="string",
                    description="ID of the draft work item to refine",
                    required=True,
                ),
                ToolParameter(
                    name="feedback",
                    type="string",
                    description="Feedback or instructions for refinement",
                    required=True,
                ),
            ],
            returns="Refined work item",
            category="generation",
            examples=[
                "Make the acceptance criteria more specific",
                "Add error handling considerations to the tasks",
            ],
        )

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute refine_item."""
        context.require_auth()

        params = self.validate_params(**kwargs)
        draft_id = params["draft_id"]
        feedback = params["feedback"]

        # Get draft from session
        draft = await self._session_manager.get_draft(
            session_id=context.session_id,
            draft_id=draft_id,
        )

        if not draft:
            return ToolResult.failure(f"Draft not found: {draft_id}")

        settings = get_settings()

        try:
            async with AzureOpenAIClient(settings.azure_openai) as client:
                refined = await client.refine_work_item(
                    work_item=draft,
                    feedback=feedback,
                )

            # Update draft in session
            await self._session_manager.update_draft(
                session_id=context.session_id,
                draft=refined,
            )

            await sync_drafts_to_conversation(context, self._session_manager)
            return ToolResult.success(
                data={
                    "id": refined.id,
                    "type": refined.work_item_type,
                    "title": refined.title,
                    "fields": refined.fields,
                },
                message=f"Refined work item: {refined.title}",
            )

        except GenerationError as e:
            logger.error(f"Refinement failed: {e}")
            return ToolResult.failure(f"Failed to refine work item: {e}")
        except OpenAIError as e:
            logger.error(f"OpenAI error: {e}")
            return ToolResult.failure(f"AI service error: {e}")


# =============================================================================
# Generate Tasks Batch Tool
# =============================================================================


class GenerateTasksBatchTool(BaseTool):
    """Generate tasks for ALL parent drafts in a single call."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="generate_tasks_batch",
            description=(
                "Generate tasks for ALL parent work items (Epics, User Stories, Features, etc.) in "
                "the current session in a single call. Use this instead of calling generate_tasks "
                "multiple times. This tool finds all non-Task drafts and generates tasks for each one."
            ),
            parameters=[
                ToolParameter(
                    name="draft_ids",
                    type="array",
                    description=(
                        "Optional list of specific draft IDs to generate tasks for. "
                        "If not provided, generates tasks for ALL parent items in the session."
                    ),
                    required=False,
                ),
            ],
            returns="Generated tasks for all parent items, linked to their parents",
            category="generation",
            examples=[
                "Generate tasks for all epics",
                "Create tasks for all user stories",
                "Break down all items into tasks",
            ],
        )

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute batch task generation for all parent drafts."""
        context.require_auth()
        context.require_project()

        params = self.validate_params(**kwargs)
        requested_ids: list[str] | None = params.get("draft_ids")

        # Get all drafts
        draft_collection = await self._session_manager.get_drafts(context.session_id)
        if not draft_collection or not draft_collection.items:
            return ToolResult.failure("No drafts found in session")

        # Get board context
        board_context = None
        if context.project_name:
            board_context = await self._session_manager.get_board_context(
                session_id=context.session_id,
                project_id=context.project_name,
            )

        # Determine which types are "Task" types (should be skipped as parents)
        task_type = "Task"
        if board_context:
            task_type = board_context.task_type

        # Find all parent items (non-Task types)
        parent_items = [
            d for d in draft_collection.items
            if d.work_item_type != task_type
            and (requested_ids is None or d.draft_id in requested_ids)
        ]

        if not parent_items:
            return ToolResult.failure("No parent items found to generate tasks for")

        settings = get_settings()
        all_generated: list[dict[str, Any]] = []
        errors: list[str] = []

        async with AzureOpenAIClient(settings.azure_openai) as client:
            for parent in parent_items:
                user_story_text = f"{parent.title}\n{parent.description or ''}"

                try:
                    result = await client.generate_tasks(
                        user_story=user_story_text,
                        board_context=board_context,
                        parent_type=parent.work_item_type,
                    )

                    for item in result.work_items:
                        item.parent_draft_id = parent.draft_id
                        await add_draft_to_session(
                            session_manager=self._session_manager,
                            session_id=context.session_id,
                            project_id=context.project_name or "unknown",
                            draft_item=item,
                        )
                        all_generated.append({
                            "draft_id": item.draft_id,
                            "type": item.work_item_type,
                            "title": item.title,
                            "description": item.description,
                            "parent_draft_id": parent.draft_id,
                            "parent_title": parent.title,
                        })

                except (ContextLengthError, GenerationError, OpenAIError) as e:
                    logger.error(f"Failed to generate tasks for '{parent.title}': {e}")
                    errors.append(f"{parent.title}: {e}")

        await sync_drafts_to_conversation(context, self._session_manager)

        if not all_generated and errors:
            return ToolResult.failure(
                f"Failed to generate tasks for all items: {'; '.join(errors)}"
            )

        return ToolResult.success(
            data={
                "total_tasks_generated": len(all_generated),
                "parent_items_processed": len(parent_items) - len(errors),
                "work_items": all_generated,
                "errors": errors if errors else None,
            },
            message=(
                f"Generated {len(all_generated)} task(s) for "
                f"{len(parent_items) - len(errors)} parent item(s)"
                + (f" ({len(errors)} failed)" if errors else "")
            ),
        )


# =============================================================================
# Tool Factory
# =============================================================================


@register_tools
def create_generation_tools(session_manager: SessionManager) -> list[BaseTool]:
    """Create generation tools with session manager dependency."""
    return [
        GenerateUserStoriesTool(session_manager),
        GenerateTasksTool(session_manager),
        GenerateTasksBatchTool(session_manager),
        GenerateBugReportTool(session_manager),
        GenerateTestCasesTool(session_manager),
        RefineItemTool(session_manager),
    ]
