"""
Azure OpenAI client for work item generation.

This module provides the AI backbone for the MCP server, using Azure OpenAI
to generate high-quality Azure DevOps work items from natural language.

Capabilities:
    - User Stories: Generate from requirements with acceptance criteria
    - Tasks: Break down user stories into actionable tasks
    - Bug Reports: Structure issue descriptions into proper bug reports
    - Refinement: Improve existing drafts based on feedback

Key Features:
    - Structured JSON output for reliable parsing
    - Context-aware generation using existing work items
    - Configurable prompts for different work item types
    - Retry logic with exponential backoff
    - Content safety filtering

Architecture:
    1. Receive generation request with requirements/context
    2. Build system prompt with work item schema
    3. Include context from existing work items (if provided)
    4. Call Azure OpenAI with JSON mode enabled
    5. Parse response into DraftWorkItem objects
    6. Return drafts for review and push

Configuration:
    Set these environment variables:
    - AZURE_OPENAI_ENDPOINT: Your Azure OpenAI resource URL
    - AZURE_OPENAI_API_KEY: API key for authentication
    - AZURE_OPENAI_DEPLOYMENT_NAME: Model deployment name (e.g., gpt-4)

Example:
    async with AzureOpenAIClient(settings) as client:
        drafts = await client.generate_user_stories(
            requirements="Build a login page with OAuth",
            context="Existing Epic: User Authentication System",
        )
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from src.models.devops import BoardContext

import httpx

from src.config.settings import AzureOpenAISettings
from src.models.work_items import DraftWorkItem

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class OpenAIError(Exception):
    """Base exception for OpenAI errors."""

    pass


class GenerationError(OpenAIError):
    """Content generation failed."""

    pass


class RateLimitError(OpenAIError):
    """Rate limit exceeded."""

    def __init__(self, retry_after: float = 60.0) -> None:
        super().__init__("Rate limit exceeded")
        self.retry_after = retry_after


class ContentFilterError(OpenAIError):
    """Content was filtered by safety system."""

    pass


class ContextLengthError(OpenAIError):
    """Input context exceeds the model's token limit."""

    def __init__(self, estimated_tokens: int, max_tokens: int) -> None:
        self.estimated_tokens = estimated_tokens
        self.max_tokens = max_tokens
        super().__init__(
            f"Your input is too large (~{estimated_tokens:,} tokens). "
            f"The model supports ~{max_tokens:,} input tokens. "
            f"Please reduce the amount of context — use fewer files, work items, "
            f"or shorten your requirements."
        )


# Token estimation: ~4 characters per token (conservative for English text + markup)
CHARS_PER_TOKEN: Final[int] = 4

# Model context window (GPT-4 Turbo / GPT-4o). Override via AZURE_OPENAI_CONTEXT_WINDOW env var.
MODEL_CONTEXT_WINDOW: Final[int] = 128_000

# Maximum characters allowed for each context source before truncation
MAX_FILE_CONTENT_CHARS: Final[int] = 300_000    # ~75K tokens — large docs
MAX_WORK_ITEM_CONTEXT_CHARS: Final[int] = 100_000  # ~25K tokens — many items
MAX_USER_CONTEXT_CHARS: Final[int] = 80_000      # ~20K tokens — typed/wiki text

# Safety margin: reserve tokens for system prompt + output
_SYSTEM_PROMPT_RESERVE: Final[int] = 2_000  # ~2K tokens for system prompt
_OUTPUT_RESERVE: Final[int] = 4_000          # matches default max_tokens


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length. ~4 chars per token for English."""
    return len(text) // CHARS_PER_TOKEN


def truncate_context(text: str, max_chars: int, label: str = "content") -> str:
    """Truncate text to max_chars with a warning appended."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Cut at last newline to avoid mid-sentence truncation
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.8:
        truncated = truncated[:last_newline]
    truncated += (
        f"\n\n[WARNING: {label} was truncated from "
        f"~{len(text):,} to ~{len(truncated):,} characters to fit within token limits. "
        f"Some details may be missing.]"
    )
    logger.warning(
        f"Truncated {label}: {len(text):,} → {len(truncated):,} chars "
        f"(~{estimate_tokens(text):,} → ~{estimate_tokens(truncated):,} tokens)"
    )
    return truncated


# =============================================================================
# Prompt Templates
# =============================================================================


class PromptTemplate(str, Enum):
    """Predefined prompt templates for work item generation."""

    USER_STORY = "user_story"
    TASK = "task"
    BUG = "bug"
    FEATURE = "feature"
    EPIC = "epic"
    TEST_CASE = "test_case"


# System prompts for different work item types (static defaults - use build_dynamic_prompt for board-aware generation)
SYSTEM_PROMPTS: Final[dict[PromptTemplate, str]] = {
    PromptTemplate.USER_STORY: """You are an expert Agile product owner and business analyst.
Your task is to generate well-structured work items from requirements.

CONTEXT GROUNDING — CRITICAL:
- Base EVERY work item strictly on information found in the provided context (files, wiki pages, work items, requirements).
- Do NOT invent features, behaviors, or requirements not present in the context.
- If the context is insufficient or vague on a topic, generate fewer items rather than fabricating details.
- When multiple context sources are provided, cross-reference them to produce accurate, consistent work items.
- Each work item should be traceable to specific details in the provided context.

IMPORTANT - Work item types: If the user explicitly requests Features, Epics, or User Stories, use exactly that type for every generated item. For mixed requests, set each item's type accordingly. Otherwise default to "User Story".

For each item, provide:
1. work_item_type: "User Story", "Epic", or "Feature" (use the explicitly requested type; otherwise "User Story")
2. title: Clear, concise title (do NOT put the word "epic" in the title for epics—use a real title)
3. description: As a [user], I want [goal], so that [benefit] (for User Story), business objectives (for Epic), or user value and scope (for Feature)
4. acceptance_criteria: Testable criteria in Given/When/Then format (for User Story; high-level outcomes for Epic or Feature)
5. story_points: Estimate (1, 2, 3, 5, 8, 13)—always provide a number so it shows on the board
6. priority: 1 (Critical), 2 (High), 3 (Medium), 4 (Low)

Output valid JSON with this exact structure:
{
  "work_items": [
    {
      "work_item_type": "User Story",
      "title": "string",
      "description": "string",
      "acceptance_criteria": "string",
      "story_points": number,
      "priority": number
    }
  ]
}

Example:
{
  "work_items": [
    {
      "work_item_type": "User Story",
      "title": "Implement SSO login via Azure AD",
      "description": "As a corporate user, I want to log in using my Azure AD credentials, so that I don't need a separate password.",
      "acceptance_criteria": "Given a user with valid Azure AD credentials\\nWhen they click 'Sign in with Microsoft'\\nThen they are authenticated and redirected to the dashboard\\n\\nGiven a user with invalid credentials\\nWhen they attempt to sign in\\nThen they see a clear error message",
      "story_points": 5,
      "priority": 2
    }
  ]
}""",

    PromptTemplate.TASK: """You are an expert software developer and technical lead.
Your task is to break down work items into concrete, implementable development tasks.

GUIDELINES:
- Every task title MUST start with an implementation verb: Implement, Create, Design, Build, Configure, Write, Set up, Integrate, Develop, Add, Deploy, Migrate, Optimize, Refactor
- Generate a balanced mix of activities: mostly Development, with Testing, Design, or Documentation where appropriate
- Each task should be 4-16 hours of focused work — not too broad, not too granular
- Descriptions should include specific technologies, approaches, and acceptance details

For each Task, provide:
1. work_item_type: Always "Task"
2. title: Actionable title starting with an implementation verb
3. description: Technical approach, technologies involved, and key acceptance details
4. remaining_work: Estimated hours (4-16 typical)
5. activity: One of: Development, Testing, Documentation, Design

Output valid JSON with this exact structure:
{
  "work_items": [
    {
      "work_item_type": "Task",
      "title": "string",
      "description": "string",
      "remaining_work": number,
      "activity": "string"
    }
  ]
}

Example:
{
  "work_items": [
    {
      "work_item_type": "Task",
      "title": "Set up OAuth 2.0 authentication middleware",
      "description": "Configure OAuth 2.0 with PKCE flow using MSAL.js. Implement token acquisition, refresh logic, and session management. Integrate with Azure AD as identity provider.",
      "remaining_work": 10,
      "activity": "Development"
    },
    {
      "work_item_type": "Task",
      "title": "Create role-based access control service",
      "description": "Build RBAC service with Admin, Manager, and Viewer roles. Implement permission-checking middleware that validates user roles on protected API endpoints.",
      "remaining_work": 8,
      "activity": "Development"
    },
    {
      "work_item_type": "Task",
      "title": "Design login and registration UI screens",
      "description": "Create wireframes and high-fidelity mockups for login, registration, password reset, and MFA enrollment. Follow existing design system and accessibility standards.",
      "remaining_work": 6,
      "activity": "Design"
    },
    {
      "work_item_type": "Task",
      "title": "Write authentication integration tests",
      "description": "Test login flow, token refresh, session expiry, invalid credentials, and role-based endpoint access. Cover both positive and negative scenarios with automated test suite.",
      "remaining_work": 6,
      "activity": "Testing"
    }
  ]
}""",

    PromptTemplate.BUG: """You are an expert QA engineer and bug analyst.
Your task is to create detailed, actionable bug reports from issue descriptions.

CONTEXT GROUNDING: Base every bug report strictly on the issue description and any provided context. Do NOT invent symptoms, steps, or behaviors. If information is missing, write "Not specified" or "Needs investigation".

For each Bug, provide:
1. work_item_type: Always "Bug"
2. title: Clear, specific description of the defect (not generic like "Bug in login"); prefix with a known product or module in square brackets when supplied (for example, [College ERP])
3. description: Use readable Markdown sections in this order: Description; Environment; Expected Result; Suggested Enhancement (only when supported by the source); Actual Result; Impact; Subtasks (only when concrete subtasks are apparent, otherwise "None")
4. repro_steps: Numbered Steps to Reproduce, preserving the user's sequence and prerequisites
5. severity: 1 (Critical - system down), 2 (High - major feature broken), 3 (Medium - workaround exists), 4 (Low - cosmetic)
6. priority: 1 (Critical), 2 (High), 3 (Medium), 4 (Low)

Output valid JSON with this exact structure:
{
  "work_items": [
    {
      "work_item_type": "Bug",
      "title": "string",
      "description": "string",
      "repro_steps": "string",
      "severity": number,
      "priority": number
    }
  ]
}

Example:
{
  "work_items": [
    {
      "work_item_type": "Bug",
      "title": "[Authentication] Session expires despite Remember Me being selected",
      "description": "## Description\\nThe session ends while Remember Me is enabled.\\n\\n## Environment\\nNot specified.\\n\\n## Expected Result\\nThe session remains active for the configured remembered-session period.\\n\\n## Suggested Enhancement\\nNeeds investigation.\\n\\n## Actual Result\\nThe session expires after approximately five minutes.\\n\\n## Impact\\nUsers must sign in repeatedly, disrupting work.\\n\\n## Subtasks\\nNone.",
      "repro_steps": "1. Navigate to the login page\\n2. Enter valid credentials\\n3. Check 'Remember Me'\\n4. Click 'Sign In'\\n5. Wait 5 minutes without activity\\n6. Attempt any action — user is redirected to login",
      "severity": 2,
      "priority": 2
    }
  ]
}""",

    PromptTemplate.FEATURE: """You are an expert product manager.
Your task is to define Features that group related User Stories and deliver cohesive business value.

CONTEXT GROUNDING: Base every feature strictly on the provided requirements and context. Do NOT invent business capabilities not mentioned in the context.

For each Feature, provide:
1. work_item_type: Always "Feature"
2. title: Clear feature name describing the capability
3. description: Business value, target users, and scope boundaries
4. acceptance_criteria: High-level success criteria for the feature
5. priority: 1 (Critical), 2 (High), 3 (Medium), 4 (Low)
6. story_points: Relative size estimate (1, 2, 3, 5, 8, 13)

Output valid JSON with this exact structure:
{
  "work_items": [
    {
      "work_item_type": "Feature",
      "title": "string",
      "description": "string",
      "acceptance_criteria": "string",
      "priority": number,
      "story_points": number
    }
  ]
}""",

    PromptTemplate.EPIC: """You are an expert product strategist.
Your task is to define Epics that represent large strategic initiatives spanning multiple features or sprints.

CONTEXT GROUNDING: Base every epic strictly on the provided requirements and context. Do NOT invent strategic initiatives not mentioned in the context.

For each Epic, provide:
1. work_item_type: Always "Epic"
2. title: Strategic initiative name (concise, no prefix like "Epic:")
3. description: Business objectives, success metrics, and high-level scope
4. acceptance_criteria: Key outcomes that define when the epic is complete
5. priority: 1 (Critical), 2 (High), 3 (Medium), 4 (Low)
6. story_points: Relative size estimate (1, 2, 3, 5, 8, 13)

Output valid JSON with this exact structure:
{
  "work_items": [
    {
      "work_item_type": "Epic",
      "title": "string",
      "description": "string",
      "acceptance_criteria": "string",
      "priority": number,
      "story_points": number
    }
  ]
}""",

    PromptTemplate.TEST_CASE: """You are a senior QA engineer. Create executable, independent test cases from the supplied story or requirement.
Ground every case in the provided source. Do not invent product behavior; call out missing details as assumptions or questions.
Cover positive, negative, boundary, validation, and permission scenarios where relevant. Avoid duplicate cases.
Return valid JSON only with this schema:
{"work_items":[{"work_item_type":"Test Case","title":"Clear scenario title","description":"Preconditions:\\n...\\nTest data:\\n...\\nSteps:\\n1. ...\\nExpected result:\\n...\\nCoverage: Positive|Negative|Boundary|Permission","acceptance_criteria":"The observable pass condition","priority":1}]}
Create 5-10 cases by default, or follow an explicit count request. Use priority 1-4 (1 highest).""",
}


# =============================================================================
# Dynamic Prompt Building (Board-Context-Aware)
# =============================================================================


def build_dynamic_user_story_prompt(board_context: BoardContext | None = None) -> str:
    """
    Build a dynamic USER_STORY prompt based on the project's board context.

    Args:
        board_context: Board context with work item types and hierarchy

    Returns:
        System prompt customized for the project's work item types
    """
    if not board_context:
        return SYSTEM_PROMPTS[PromptTemplate.USER_STORY]

    allowed_types = board_context.get_allowed_types_string()
    hierarchy = board_context.get_hierarchy_description()
    default_type = board_context.default_requirement_type
    epic_type = board_context.epic_type or "Epic"
    story_points_field = board_context.story_points_field

    # Determine field name for display
    effort_field_name = "story_points" if story_points_field == "story_points" else "effort"

    return f"""You are an expert Agile product owner and business analyst.
Your task is to generate well-structured work items from requirements.

CONTEXT GROUNDING — CRITICAL:
- Base EVERY work item strictly on information found in the provided context (files, wiki pages, work items, requirements).
- Do NOT invent features, behaviors, or requirements not present in the context.
- If the context is insufficient or vague on a topic, generate fewer items rather than fabricating details.
- When multiple context sources are provided, cross-reference them to produce accurate, consistent work items.
- Each work item should be traceable to specific details in the provided context.

PROJECT BOARD STRUCTURE:
- Available work item types: {allowed_types}
- Hierarchy: {hierarchy}
- Default backlog type: {default_type}

IMPORTANT - Work item types:
- Use work_item_type from the available types: {allowed_types}
- If the user asks for a mix (e.g. "3 {default_type.lower()}s and 1 {epic_type.lower()}"), output each item with "work_item_type" set appropriately.
- Otherwise default to "{default_type}" for backlog items.
- Use "{epic_type}" only when the user explicitly asks for {epic_type.lower()}s or portfolio items.

For each item, provide:
1. work_item_type: MUST be one of: {allowed_types}
2. title: Clear, concise title (do NOT prefix with the type name)
3. description: As a [user], I want [goal], so that [benefit] (for {default_type}) or business objectives (for {epic_type})
4. acceptance_criteria: Testable criteria in Given/When/Then format (for {default_type}; optional for {epic_type})
5. {effort_field_name}: Estimate (1, 2, 3, 5, 8, 13)—always provide a number so it shows on the board
6. priority: 1 (Critical), 2 (High), 3 (Medium), 4 (Low)

Output valid JSON with this exact structure:
{{
  "work_items": [
    {{
      "work_item_type": "{default_type}",
      "title": "string",
      "description": "string",
      "acceptance_criteria": "string",
      "{effort_field_name}": number,
      "priority": number
    }}
  ]
}}

Example:
{{
  "work_items": [
    {{
      "work_item_type": "{default_type}",
      "title": "Implement SSO login via Azure AD",
      "description": "As a corporate user, I want to log in using my Azure AD credentials, so that I don't need a separate password.",
      "acceptance_criteria": "Given a user with valid Azure AD credentials\\nWhen they click 'Sign in with Microsoft'\\nThen they are authenticated and redirected to the dashboard",
      "{effort_field_name}": 5,
      "priority": 2
    }}
  ]
}}"""


def build_dynamic_task_prompt(board_context: BoardContext | None = None) -> str:
    """
    Build a dynamic TASK prompt based on the project's board context.

    Args:
        board_context: Board context with work item types and hierarchy

    Returns:
        System prompt customized for the project's task generation
    """
    if not board_context:
        return SYSTEM_PROMPTS[PromptTemplate.TASK]

    task_type = board_context.task_type
    parent_types = ", ".join(board_context.parent_types_for_tasks)

    return f"""You are an expert software developer and technical lead.
Your task is to break down work items ({parent_types}) into concrete, implementable development tasks.

GUIDELINES:
- Every task title MUST start with an implementation verb: Implement, Create, Design, Build, Configure, Write, Set up, Integrate, Develop, Add, Deploy, Migrate, Optimize, Refactor
- Generate a balanced mix of activities: mostly Development, with Testing, Design, or Documentation where appropriate
- Each task should be 4-16 hours of focused work — not too broad, not too granular
- Descriptions should include specific technologies, approaches, and acceptance details

PROJECT BOARD STRUCTURE:
- Task type: {task_type}
- Parent types: {parent_types}

For each {task_type}, provide:
1. work_item_type: Always "{task_type}"
2. title: Actionable title starting with an implementation verb
3. description: Technical approach, technologies involved, and key acceptance details
4. remaining_work: Estimated hours (4-16 typical)
5. activity: One of: Development, Testing, Documentation, Design

Output valid JSON with this exact structure:
{{
  "work_items": [
    {{
      "work_item_type": "{task_type}",
      "title": "string",
      "description": "string",
      "remaining_work": number,
      "activity": "string"
    }}
  ]
}}

Example:
{{
  "work_items": [
    {{
      "work_item_type": "{task_type}",
      "title": "Build REST API endpoints for inventory CRUD operations",
      "description": "Create GET/POST/PUT/DELETE endpoints for inventory items. Include pagination, filtering by category/warehouse, and input validation. Use repository pattern for data access.",
      "remaining_work": 10,
      "activity": "Development"
    }},
    {{
      "work_item_type": "{task_type}",
      "title": "Implement real-time stock level notification service",
      "description": "Build event-driven service that monitors inventory thresholds and sends alerts via email and in-app notifications when stock drops below configured minimums.",
      "remaining_work": 8,
      "activity": "Development"
    }},
    {{
      "work_item_type": "{task_type}",
      "title": "Write end-to-end tests for inventory management workflow",
      "description": "Test complete inventory lifecycle: create item, update stock, trigger low-stock alert, reorder flow. Cover edge cases like concurrent updates and zero-stock scenarios.",
      "remaining_work": 6,
      "activity": "Testing"
    }}
  ]
}}"""


def build_dynamic_bug_prompt(board_context: BoardContext | None = None) -> str:
    """
    Build a dynamic BUG prompt based on the project's board context.

    Args:
        board_context: Board context with work item types and hierarchy

    Returns:
        System prompt customized for the project's bug type
    """
    if not board_context:
        return SYSTEM_PROMPTS[PromptTemplate.BUG]

    bug_type = board_context.bug_type

    return f"""You are an expert QA engineer and bug analyst.
Your task is to create detailed, actionable bug reports from issue descriptions.

CONTEXT GROUNDING: Base every bug report strictly on the issue description and any provided context. Do NOT invent symptoms, steps, or behaviors not described. If details are missing, note them as "needs investigation" rather than guessing.

PROJECT BOARD STRUCTURE:
- Bug type: {bug_type}

For each {bug_type}, provide:
1. work_item_type: Always "{bug_type}"
2. title: Clear, specific description of the defect (not generic like "Bug in login")
3. description: Include Expected Behavior, Actual Behavior, and Impact sections
4. repro_steps: Numbered steps to reproduce, starting from a clean state
5. severity: 1 (Critical - system down), 2 (High - major feature broken), 3 (Medium - workaround exists), 4 (Low - cosmetic)
6. priority: 1 (Critical), 2 (High), 3 (Medium), 4 (Low)

Output valid JSON with this exact structure:
{{
  "work_items": [
    {{
      "work_item_type": "{bug_type}",
      "title": "string",
      "description": "string",
      "repro_steps": "string",
      "severity": number,
      "priority": number
    }}
  ]
}}

Example:
{{
  "work_items": [
    {{
      "work_item_type": "{bug_type}",
      "title": "Login session expires after 5 minutes despite 'Remember Me' checked",
      "description": "Expected: Session persists for 30 days when Remember Me is selected.\\nActual: Session expires after ~5 minutes regardless of the checkbox state.\\nImpact: Users must re-authenticate frequently, disrupting workflow.",
      "repro_steps": "1. Navigate to the login page\\n2. Enter valid credentials\\n3. Check 'Remember Me'\\n4. Click 'Sign In'\\n5. Wait 5 minutes without activity\\n6. Attempt any action — user is redirected to login",
      "severity": 2,
      "priority": 2
    }}
  ]
}}"""


# =============================================================================
# Azure OpenAI Client
# =============================================================================


@dataclass
class GenerationResult:
    """Result of AI generation."""

    work_items: list[DraftWorkItem]
    prompt_tokens: int
    completion_tokens: int
    model: str


# Two-part filter for filler/invalid tasks.
# A task is invalid if its title contains BOTH a planning verb AND a process noun.
# This catches variations like "Break down business requirements" or
# "Clarify acceptance criteria for Epic UUID" that simple substring matching misses.
_PLANNING_VERBS: Final[list[str]] = [
    "clarify", "gather", "break down", "assess", "investigate",
    "request clarification", "analyze", "review", "define", "refine",
    "determine", "identify", "evaluate", "consult", "discuss",
    "schedule", "coordinate", "prioritize", "estimate",
]
_PROCESS_NOUNS: Final[list[str]] = [
    "requirements", "scope", "criteria", "specifications",
    "details", "information", "constraints", "stakeholder",
    "epic", "user stories", "user story", "feature",
]


def _filter_invalid_tasks(work_items: list[DraftWorkItem]) -> list[DraftWorkItem]:
    """
    Remove filler tasks whose titles match planning/analysis patterns.

    A task is invalid when its title contains a planning verb paired with
    a process noun (e.g. "Clarify acceptance criteria", "Break down
    business requirements"). This two-part check catches creative
    variations that simple substring matching would miss.
    """
    valid = []
    for item in work_items:
        title_lower = item.title.lower()
        has_planning_verb = any(v in title_lower for v in _PLANNING_VERBS)
        has_process_noun = any(n in title_lower for n in _PROCESS_NOUNS)
        if has_planning_verb and has_process_noun:
            logger.warning(f"Filtered invalid task: '{item.title}'")
            continue
        valid.append(item)
    return valid


class AzureOpenAIClient:
    """
    Azure OpenAI client for work item generation.

    Uses GPT-4 with structured output (JSON mode) for reliable parsing.
    Includes retry logic and rate limit handling.

    Usage:
        async with AzureOpenAIClient(settings) as client:
            result = await client.generate_user_stories(requirements)
    """

    def __init__(self, settings: AzureOpenAISettings) -> None:
        """
        Initialize Azure OpenAI client.

        Args:
            settings: Azure OpenAI settings
        """
        self._settings = settings
        self._provider = os.getenv("AI_PROVIDER", "azure").strip().lower()
        if self._provider == "openai":
            self._endpoint = (os.getenv("OPENAI_API_BASE_URL") or "https://api.openai.com/v1").rstrip("/") + "/"
            self._api_key = os.getenv("OPENAI_API_KEY", "")
            self._deployment = os.getenv("OPENAI_MODEL", "gpt-6.2-sol")
            self._api_version = ""
        else:
            self._provider = "azure"
            self._endpoint = settings.endpoint
            self._api_key = settings.api_key.get_secret_value()
            self._deployment = settings.deployment_name
            self._api_version = settings.api_version

        # HTTP client (initialized in __aenter__)
        self._client: httpx.AsyncClient | None = None

        logger.debug(
            "Azure OpenAI client initialized",
            extra={"deployment": self._deployment},
        )

    async def __aenter__(self) -> AzureOpenAIClient:
        """Async context manager entry."""
        if not self._api_key:
            provider_setting = "OPENAI_API_KEY" if self._provider == "openai" else "AZURE_OPENAI_API_KEY"
            raise GenerationError(f"{provider_setting} must be configured for the selected AI provider")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0),  # Long timeout for generation
            headers={
                **({"Authorization": f"Bearer {self._api_key}"} if self._provider == "openai" else {"api-key": self._api_key}),
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
    # Core Generation Method
    # =========================================================================

    async def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Generate content using Azure OpenAI.

        Args:
            system_prompt: System message for context
            user_prompt: User message with requirements
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Parsed JSON response

        Raises:
            GenerationError: If generation fails
            RateLimitError: If rate limited
            ContentFilterError: If content filtered
        """
        client = self._get_client()

        url = (
            f"{self._endpoint}chat/completions"
            if self._provider == "openai"
            else f"{self._endpoint}openai/deployments/{self._deployment}/chat/completions?api-version={self._api_version}"
        )

        output_tokens = max_tokens or self._settings.max_tokens

        # Pre-validate: estimate input tokens and check against model limit
        input_chars = len(system_prompt) + len(user_prompt)
        estimated_input_tokens = input_chars // CHARS_PER_TOKEN
        max_input_tokens = MODEL_CONTEXT_WINDOW - output_tokens
        if estimated_input_tokens > max_input_tokens:
            raise ContextLengthError(
                estimated_tokens=estimated_input_tokens,
                max_tokens=max_input_tokens,
            )

        logger.info(
            f"OpenAI request: ~{estimated_input_tokens:,} input tokens, "
            f"{output_tokens:,} output tokens reserved, "
            f"{MODEL_CONTEXT_WINDOW:,} context window"
        )

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature or self._settings.temperature,
            "max_completion_tokens" if self._provider == "openai" and self._deployment.startswith("gpt-6") else "max_tokens": output_tokens,
            "response_format": {"type": "json_object"},
        }
        if self._provider == "openai":
            payload["model"] = self._deployment
            if self._deployment == "gpt-6.2-sol":
                payload.pop("temperature", None)
                payload["reasoning_effort"] = "medium"

        try:
            response = await client.post(url, json=payload)

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", 60))
                raise RateLimitError(retry_after)

            # Handle 400 errors — check specific error codes
            if response.status_code == 400:
                data = response.json()
                error_str = str(data)
                if "context_length_exceeded" in error_str:
                    raise ContextLengthError(
                        estimated_tokens=estimated_input_tokens,
                        max_tokens=max_input_tokens,
                    )
                if "content_filter" in error_str:
                    raise ContentFilterError("Content was filtered by safety system")

            # Handle other errors
            if response.status_code != 200:
                raise GenerationError(
                    f"Generation failed: {response.status_code} - {response.text}"
                )

            data = response.json()

            # Extract content
            content = data["choices"][0]["message"]["content"]
            logger.info(f"Azure OpenAI raw response: {content}")
            parsed_content = json.loads(content)
            logger.info(f"Parsed content keys: {parsed_content.keys() if isinstance(parsed_content, dict) else 'not a dict'}")
            return {
                "content": parsed_content,
                "usage": data.get("usage", {}),
                "model": data.get("model", self._deployment),
            }

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse OpenAI response: {e}")
            raise GenerationError(f"Failed to parse response: {e}") from e
        except httpx.HTTPError as e:
            logger.error(f"HTTP error during generation: {e}")
            raise GenerationError(f"HTTP error: {e}") from e


    # =========================================================================
    # Work Item Generation Methods
    # =========================================================================

    def _parse_work_items(
        self,
        data: dict[str, Any],
        work_item_type: str,
    ) -> list[DraftWorkItem]:
        """Parse generated work items from AI response."""
        import uuid

        items = []
        for item_data in data.get("work_items", []):
            # story_points or effort (LLM may use either)
            sp = item_data.get("story_points")
            if sp is None:
                sp = item_data.get("effort")

            # Build description - append repro_steps for bugs if present
            description = item_data.get("description", "")
            repro_steps = item_data.get("repro_steps")
            if repro_steps:
                description = f"{description}\n\n<b>Repro Steps:</b>\n{repro_steps}" if description else f"<b>Repro Steps:</b>\n{repro_steps}"

            # Create DraftWorkItem with proper field mapping
            draft_item = DraftWorkItem(
                draft_id=str(uuid.uuid4()),
                work_item_type=item_data.get("work_item_type", work_item_type),
                title=item_data.get("title", "Untitled"),
                description=description,
                acceptance_criteria=item_data.get("acceptance_criteria"),
                priority=item_data.get("priority"),
                story_points=float(sp) if sp is not None else None,
            )

            # Store bug-specific severity in custom_fields
            severity = item_data.get("severity")
            if severity is not None:
                draft_item.custom_fields["Microsoft.VSTS.Common.Severity"] = str(severity)

            items.append(draft_item)
            logger.info(f"Parsed work item: {draft_item.title}")
        return items

    async def generate_user_stories(
        self,
        requirements: str,
        context: str | None = None,
        board_context: BoardContext | None = None,
        count: int | None = None,
    ) -> GenerationResult:
        """
        Generate User Stories from requirements.

        Args:
            requirements: Natural language requirements
            context: Additional context (project info, constraints)
            board_context: Board context with work item types (makes prompts dynamic)

        Returns:
            GenerationResult with draft work items
        """
        user_prompt = f"Requirements:\n{requirements}"
        if count is not None:
            user_prompt = (
                f"Generate exactly {count} work item(s).\n\n{user_prompt}"
            )
        if context:
            user_prompt = f"Context:\n{context}\n\n{user_prompt}"

        # Use dynamic prompt if board context is available
        system_prompt = build_dynamic_user_story_prompt(board_context)

        result = await self._generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # Use default type from board context if available
        default_type = board_context.default_requirement_type if board_context else "User Story"
        work_items = self._parse_work_items(result["content"], default_type)

        return GenerationResult(
            work_items=work_items,
            prompt_tokens=result["usage"].get("prompt_tokens", 0),
            completion_tokens=result["usage"].get("completion_tokens", 0),
            model=result["model"],
        )

    async def generate_tasks(
        self,
        user_story: str,
        context: str | None = None,
        board_context: BoardContext | None = None,
        parent_type: str | None = None,
    ) -> GenerationResult:
        """
        Generate Tasks from a User Story, Epic, or Feature.

        Args:
            user_story: Parent work item text/description (user story, epic, etc.)
            context: Additional context
            board_context: Board context with work item types (makes prompts dynamic)
            parent_type: Explicit parent type (Epic, User Story, Feature) when known.
                         Used for prompt labeling; overrides board_context default.
        Returns:
            GenerationResult with draft work items
        """
        # Use explicit parent_type when provided (e.g. Epic), else board context default
        if parent_type:
            prompt_parent_type = parent_type
        else:
            prompt_parent_type = (
                board_context.default_requirement_type if board_context else "User Story"
            )

        # Build a rich user prompt with explicit instructions embedded.
        # Models prioritize user-prompt instructions over system-prompt ones,
        # so the key directive must live here to prevent filler-task output.
        user_prompt = (
            f"Generate 4-8 concrete implementation tasks for the following {prompt_parent_type}. "
            f"Each task title MUST start with a verb like Implement, Create, Design, Build, "
            f"Configure, Write, Set up, Integrate, or Deploy. "
            f"Include a mix of Development, Testing, and Design activities. "
            f"Do NOT output planning or analysis tasks — only real implementation work.\n\n"
            f"{prompt_parent_type}:\n{user_story}"
        )
        if context:
            user_prompt = f"Context:\n{context}\n\n{user_prompt}"

        # Use dynamic prompt if board context is available
        system_prompt = build_dynamic_task_prompt(board_context)
        task_type = board_context.task_type if board_context else "Task"

        # Use lower temperature for task generation — higher values cause
        # the model to "creatively" output filler tasks like "Clarify requirements".
        result = await self._generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
        )

        work_items = self._parse_work_items(result["content"], task_type)
        work_items = _filter_invalid_tasks(work_items)

        # If every task was filtered out, retry once at near-zero temperature
        if not work_items:
            logger.warning("All tasks filtered as invalid — retrying with temperature=0.1")
            result = await self._generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
            )
            work_items = self._parse_work_items(result["content"], task_type)
            work_items = _filter_invalid_tasks(work_items)

        return GenerationResult(
            work_items=work_items,
            prompt_tokens=result["usage"].get("prompt_tokens", 0),
            completion_tokens=result["usage"].get("completion_tokens", 0),
            model=result["model"],
        )

    async def generate_bug_report(
        self,
        issue_description: str,
        context: str | None = None,
        board_context: BoardContext | None = None,
    ) -> GenerationResult:
        """Generate Bug report from issue description."""
        user_prompt = f"Issue:\n{issue_description}"
        if context:
            user_prompt = f"Context:\n{context}\n\n{user_prompt}"

        system_prompt = build_dynamic_bug_prompt(board_context)

        result = await self._generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        bug_type = board_context.bug_type if board_context else "Bug"
        work_items = self._parse_work_items(result["content"], bug_type)

        return GenerationResult(
            work_items=work_items,
            prompt_tokens=result["usage"].get("prompt_tokens", 0),
            completion_tokens=result["usage"].get("completion_tokens", 0),
            model=result["model"],
        )

    async def generate_test_cases(
        self,
        requirement: str,
        context: str | None = None,
        count: int | None = None,
    ) -> GenerationResult:
        """Generate independently reviewable QA test case drafts."""
        user_prompt = f"Requirement or user story:\n{requirement}"
        if count is not None:
            user_prompt = f"Generate exactly {count} test cases.\n\n{user_prompt}"
        if context:
            user_prompt = f"Additional context:\n{context}\n\n{user_prompt}"
        result = await self._generate(
            system_prompt=PromptTemplate.TEST_CASE.value,
            user_prompt=user_prompt,
            temperature=0.2,
        )
        work_items = self._parse_work_items(result["content"], "Test Case")
        return GenerationResult(
            work_items=work_items,
            prompt_tokens=result["usage"].get("prompt_tokens", 0),
            completion_tokens=result["usage"].get("completion_tokens", 0),
            model=result["model"],
        )

    async def refine_work_item(
        self,
        work_item: DraftWorkItem,
        feedback: str,
    ) -> DraftWorkItem:
        """
        Refine a work item based on feedback.

        Args:
            work_item: Original draft work item
            feedback: User feedback for refinement

        Returns:
            Refined draft work item
        """
        system_prompt = f"""You are refining a {work_item.work_item_type}.
Apply the user's feedback to improve the work item. Preserve all fields not explicitly mentioned in the feedback.

Return the complete refined work item as valid JSON with this exact structure:
{{
  "title": "string",
  "description": "string",
  "acceptance_criteria": "string or null",
  "story_points": number or null,
  "priority": number or null (1=Critical, 2=High, 3=Medium, 4=Low)
}}"""

        # Build user prompt with all current fields
        ac = work_item.acceptance_criteria or "None"
        sp = work_item.story_points if work_item.story_points is not None else "None"
        pr = work_item.priority if work_item.priority is not None else "None"
        desc = work_item.description or ""

        user_prompt = f"""Original Work Item:
Title: {work_item.title}
Description: {desc}
Acceptance Criteria: {ac}
Story Points: {sp}
Priority: {pr}

Feedback: {feedback}"""

        result = await self._generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # Parse refined item - update all fields from response
        refined_data = result["content"]
        work_item.title = refined_data.get("title", work_item.title)
        work_item.fields["System.Title"] = work_item.title
        if "description" in refined_data:
            work_item.description = refined_data["description"]
            work_item.fields["System.Description"] = refined_data["description"]
        if "acceptance_criteria" in refined_data and refined_data["acceptance_criteria"] is not None:
            work_item.acceptance_criteria = refined_data["acceptance_criteria"]
            work_item.fields["Microsoft.VSTS.Common.AcceptanceCriteria"] = refined_data[
                "acceptance_criteria"
            ]
        if "story_points" in refined_data and refined_data["story_points"] is not None:
            work_item.story_points = float(refined_data["story_points"])
        if "priority" in refined_data and refined_data["priority"] is not None:
            work_item.priority = int(refined_data["priority"])

        return work_item
