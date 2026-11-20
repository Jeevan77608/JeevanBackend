"""MCP tools for the connected user's Jira Cloud account."""

from __future__ import annotations

import logging
import re
from typing import Any

from src.config.settings import get_settings
from src.integrations.jira import JiraCloudClient, JiraIntegrationError
from src.mcp_tools.base import (
    BaseTool,
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from src.mcp_tools.conversations import (
    ensure_session_drafts_from_conversation,
    restore_session_drafts_from_conversation,
    sync_drafts_to_conversation,
)
from src.mcp_tools.registry import register_tools
from src.models.user import SessionContext
from src.session.manager import SessionManager

logger = logging.getLogger(__name__)

_PROJECT_KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,254}")
_ISSUE_KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,254}-\d+")
_ISSUE_TYPE_ALIASES = {
    "user story": ("Story", "User Story", "Task", "Issue"),
    "story": ("Story", "User Story", "Task", "Issue"),
    "task": ("Task",),
    "bug": ("Bug", "Defect", "Issue"),
    "epic": ("Epic", "Feature"),
    "feature": ("Feature", "Epic"),
    "issue": ("Issue", "Task", "Story"),
}


class JiraTool(BaseTool):
    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager
        self._client = JiraCloudClient(get_settings())

    async def _session(self, context: ToolContext) -> SessionContext:
        return await self._session_manager.get_session(context.session_id)

    async def _request(
        self,
        context: ToolContext,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        session = await self._session(context)
        return await self._client.request(
            session,
            self._session_manager,
            method,
            path,
            params=params,
            json_body=json_body,
        )

    @staticmethod
    def _project_key(context: ToolContext) -> str:
        if not context.project_id or not _PROJECT_KEY_RE.fullmatch(context.project_id):
            raise ValueError("Select a Jira project first using select_jira_project.")
        return context.project_id


class ListJiraProjectsTool(JiraTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_jira_projects",
            description="List Jira projects available to the connected Atlassian account.",
            parameters=[],
            returns="Jira projects with project keys and names",
            category="jira",
        )

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        try:
            session = await self._session(context)
            connection = session.integration_tokens.get("jira", {})
            site_url = str(connection.get("site_url", "")).rstrip("/")
            data = await self._request(
                context, "GET", "project/search",
                params={"maxResults": 100, "orderBy": "name"},
            )
            projects = [
                {
                    "id": item.get("key"),
                    "key": item.get("key"),
                    "name": item.get("name"),
                    "description": item.get("description"),
                    "url": f"{site_url}/browse/{item.get('key')}",
                    "state": "active",
                }
                for item in data.get("values", [])
                if isinstance(item, dict)
            ]
            return ToolResult.success(
                data={"organization_url": site_url, "projects": projects},
                message=f"Found {len(projects)} Jira project(s).",
            )
        except JiraIntegrationError as error:
            return ToolResult.failure(str(error))


class SelectJiraProjectTool(JiraTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="select_jira_project",
            description="Select a Jira project by its project key for subsequent issue operations.",
            parameters=[
                ToolParameter(
                    name="project_key",
                    type="string",
                    description="Jira project key, such as MDP or WAD.",
                    required=True,
                )
            ],
            returns="Selected Jira project key and name",
            category="jira",
        )

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        params = self.validate_params(**kwargs)
        project_key = str(params["project_key"]).strip()
        if not _PROJECT_KEY_RE.fullmatch(project_key):
            return ToolResult.failure("Invalid Jira project key.")
        try:
            project = await self._request(context, "GET", f"project/{project_key}")
            session = await self._session(context)
            session.selected_project = project["key"]
            session.metadata["project_name"] = project["name"]
            await self._session_manager.update_session(session)
            site_url = str(session.integration_tokens.get("jira", {}).get("site_url", "")).rstrip("/")
            return ToolResult.success(
                data={
                    "id": project["key"],
                    "key": project["key"],
                    "name": project["name"],
                    "description": project.get("description"),
                    "url": f"{site_url}/browse/{project['key']}",
                    "state": "active",
                    "organization_url": site_url,
                },
                message=f"Selected Jira project {project['key']} — {project['name']}.",
            )
        except (JiraIntegrationError, KeyError) as error:
            logger.warning("Could not select Jira project: %s", type(error).__name__)
            return ToolResult.failure(
                str(error) if isinstance(error, JiraIntegrationError) else "Jira returned invalid project data."
            )


class GetJiraIssueTypesTool(JiraTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_jira_issue_types",
            description="List issue types available in the selected Jira project.",
            parameters=[],
            returns="Issue type IDs and names",
            category="jira",
        )

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        try:
            project_key = self._project_key(context)
            project = await self._request(context, "GET", f"project/{project_key}")
            issue_types = project.get("issueTypes", [])
            return ToolResult.success(
                data=[
                    {"id": item.get("id"), "name": item.get("name"), "subtask": item.get("subtask", False)}
                    for item in issue_types
                    if isinstance(item, dict)
                ],
                message=f"Found {len(issue_types)} issue type(s).",
            )
        except (JiraIntegrationError, ValueError) as error:
            return ToolResult.failure(str(error))


class SearchJiraIssuesTool(JiraTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="search_jira_issues",
            description="Search issues in the selected Jira project by exact issue key, type, status, or summary text.",
            parameters=[
                ToolParameter(name="issue_key", type="string", description="Exact Jira issue key.", required=False),
                ToolParameter(name="issue_type", type="string", description="Jira issue type to filter by.", required=False),
                ToolParameter(name="status", type="string", description="Jira status to filter by.", required=False),
                ToolParameter(name="summary_contains", type="string", description="Text to search in issue summaries.", required=False),
            ],
            returns="Matching Jira issues with keys, summaries, statuses, and issue types",
            category="jira",
        )

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        params = self.validate_params(**kwargs)
        try:
            project_key = self._project_key(context)
            clauses = [f"project = {project_key}"]
            issue_key = params.get("issue_key")
            if issue_key:
                issue_key = str(issue_key).strip().upper()
                if not _ISSUE_KEY_RE.fullmatch(issue_key):
                    return ToolResult.failure("Invalid Jira issue key.")
                clauses.append(f'key = "{issue_key}"')
            for name, value in (("issuetype", params.get("issue_type")), ("status", params.get("status"))):
                if value:
                    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
                    clauses.append(f'{name} = "{escaped}"')
            summary = params.get("summary_contains")
            if summary:
                escaped = str(summary).replace("\\", "\\\\").replace('"', '\\"')
                clauses.append(f'summary ~ "{escaped}"')
            jql = " AND ".join(clauses) + " ORDER BY updated DESC"
            data = await self._request(
                context,
                "GET",
                "search/jql",
                params={
                    "jql": jql,
                    "maxResults": 50,
                    "fields": "summary,description,status,issuetype,project,updated",
                },
            )
            site_url = await self._site_url(context)
            issues = [
                {
                    "key": issue.get("key"),
                    "summary": issue.get("fields", {}).get("summary"),
                    "description": _adf_to_plain_text(issue.get("fields", {}).get("description")),
                    "status": issue.get("fields", {}).get("status", {}).get("name"),
                    "issue_type": issue.get("fields", {}).get("issuetype", {}).get("name"),
                    "url": f"{site_url}/browse/{issue.get('key')}",
                }
                for issue in data.get("issues", [])
            ]
            return ToolResult.success(
                data={"issues": issues, "count": len(issues), "jql": jql},
                message=f"Found {len(issues)} Jira issue(s).",
            )
        except (JiraIntegrationError, ValueError) as error:
            return ToolResult.failure(str(error))

    async def _site_url(self, context: ToolContext) -> str:
        session = await self._session(context)
        connection = session.integration_tokens.get("jira", {})
        return str(connection.get("site_url", "")).rstrip("/")


class GetJiraIssueTool(JiraTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_jira_issue",
            description="Get details of a Jira issue by its issue key, such as MDP-123.",
            parameters=[
                ToolParameter(name="issue_key", type="string", description="Jira issue key.", required=True)
            ],
            returns="Issue details and description",
            category="jira",
        )

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        params = self.validate_params(**kwargs)
        issue_key = str(params["issue_key"]).strip().upper()
        if not _ISSUE_KEY_RE.fullmatch(issue_key):
            return ToolResult.failure("Invalid Jira issue key.")
        try:
            issue = await self._request(
                context,
                "GET",
                f"issue/{issue_key}",
                params={"fields": "summary,description,status,issuetype,project,assignee,updated"},
            )
            fields = issue.get("fields", {})
            return ToolResult.success(
                data={
                    "key": issue.get("key"),
                    "summary": fields.get("summary"),
                    "description": fields.get("description"),
                    "status": fields.get("status", {}).get("name"),
                    "issue_type": fields.get("issuetype", {}).get("name"),
                    "project": fields.get("project", {}).get("key"),
                    "assignee": fields.get("assignee", {}).get("displayName"),
                    "updated": fields.get("updated"),
                },
                message=f"Retrieved Jira issue {issue_key}.",
            )
        except JiraIntegrationError as error:
            return ToolResult.failure(str(error))


class UpdateJiraIssueTool(JiraTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="update_jira_issue",
            description="Update a Jira issue summary or description.",
            parameters=[
                ToolParameter(name="issue_key", type="string", description="Jira issue key.", required=True),
                ToolParameter(name="summary", type="string", description="New issue summary.", required=False),
                ToolParameter(name="description", type="string", description="New issue description.", required=False),
            ],
            returns="Updated Jira issue key",
            category="jira",
        )

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        params = self.validate_params(**kwargs)
        issue_key = str(params["issue_key"]).strip().upper()
        if not _ISSUE_KEY_RE.fullmatch(issue_key):
            return ToolResult.failure("Invalid Jira issue key.")
        fields: dict[str, Any] = {}
        if params.get("summary") is not None:
            fields["summary"] = params["summary"]
        if params.get("description") is not None:
            fields["description"] = _adf_text(str(params["description"]))
        if not fields:
            return ToolResult.failure("Provide a new summary or description.")
        try:
            await self._request(
                context, "PUT", f"issue/{issue_key}", json_body={"fields": fields}
            )
            return ToolResult.success(
                data={"key": issue_key},
                message=f"Updated Jira issue {issue_key}.",
            )
        except JiraIntegrationError as error:
            return ToolResult.failure(str(error))


class AddJiraCommentTool(JiraTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="add_jira_comment",
            description="Add a comment to a Jira issue.",
            parameters=[
                ToolParameter(name="issue_key", type="string", description="Jira issue key.", required=True),
                ToolParameter(name="text", type="string", description="Comment text.", required=True),
            ],
            returns="Created comment information",
            category="jira",
        )

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        params = self.validate_params(**kwargs)
        issue_key = str(params["issue_key"]).strip().upper()
        if not _ISSUE_KEY_RE.fullmatch(issue_key):
            return ToolResult.failure("Invalid Jira issue key.")
        try:
            result = await self._request(
                context,
                "POST",
                f"issue/{issue_key}/comment",
                json_body={"body": _adf_text(str(params["text"]))},
            )
            return ToolResult.success(
                data={"id": result.get("id"), "issue_key": issue_key},
                message=f"Added a comment to {issue_key}.",
            )
        except JiraIntegrationError as error:
            return ToolResult.failure(str(error))


class PushToJiraTool(JiraTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="push_to_jira",
            description="Create the current reviewed work-item drafts as issues in the selected Jira project.",
            parameters=[
                ToolParameter(
                    name="draft_ids",
                    type="array",
                    description="Optional IDs of drafts to create. If omitted, creates all current drafts.",
                    required=False,
                ),
                ToolParameter(
                    name="additional_tags",
                    type="array",
                    description="Optional Jira labels to apply to every created issue.",
                    required=False,
                ),
            ],
            returns="Created Jira issue keys and any draft-specific failures",
            category="jira",
        )

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        try:
            project_key = self._project_key(context)
        except ValueError as error:
            return ToolResult.failure(str(error))
        params = self.validate_params(**kwargs)
        additional_tags = params.get("additional_tags") or []
        if context.conversation_id:
            await ensure_session_drafts_from_conversation(context, self._session_manager)
        draft_collection = await self._session_manager.get_drafts(context.session_id)
        drafts = draft_collection.items if draft_collection else []
        if not drafts and context.conversation_id:
            await restore_session_drafts_from_conversation(context, self._session_manager)
            draft_collection = await self._session_manager.get_drafts(context.session_id)
            drafts = draft_collection.items if draft_collection else []
        selected_ids = params.get("draft_ids")
        if selected_ids:
            drafts = [draft for draft in drafts if draft.draft_id in selected_ids]
        if not drafts:
            return ToolResult.failure("There are no drafts to create in Jira.")

        try:
            project = await self._request(context, "GET", f"project/{project_key}")
            available_types = [
                item["name"]
                for item in project.get("issueTypes", [])
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            ]
            if not available_types:
                return ToolResult.failure(
                    f"Jira did not return issue types for project {project_key}."
                )
        except JiraIntegrationError as error:
            return ToolResult.failure(str(error))

        created: list[dict[str, Any]] = []
        failed: list[dict[str, str]] = []
        for draft in drafts:
            issue_type = _resolve_issue_type(draft.work_item_type, available_types)
            if not issue_type:
                failed.append(
                    {
                        "draft_id": draft.draft_id,
                        "title": draft.title,
                        "error": f"No compatible Jira issue type for {draft.work_item_type}.",
                    }
                )
                continue
            description = draft.description or ""
            if draft.acceptance_criteria:
                description = f"{description}\n\nAcceptance criteria:\n{draft.acceptance_criteria}".strip()
            fields: dict[str, Any] = {
                "project": {"key": project_key},
                "summary": draft.title,
                "issuetype": {"name": issue_type},
            }
            if description:
                fields["description"] = _adf_text(description)
            labels = list(draft.tags)
            for tag in additional_tags:
                clean_tag = str(tag).strip()
                if clean_tag and clean_tag not in labels:
                    labels.append(clean_tag)
            if labels:
                fields["labels"] = labels
            try:
                result = await self._request(
                    context, "POST", "issue", json_body={"fields": fields}
                )
                key = result.get("key")
                if not isinstance(key, str):
                    failed.append(
                        {
                            "draft_id": draft.draft_id,
                            "title": draft.title,
                            "error": "Jira did not return the created issue key.",
                        }
                    )
                    continue
                session = await self._session(context)
                site_url = str(session.integration_tokens["jira"].get("site_url", "")).rstrip("/")
                created.append(
                    {
                        "draft_id": draft.draft_id,
                        "key": key,
                        "type": issue_type,
                        "title": draft.title,
                        "url": f"{site_url}/browse/{key}",
                    }
                )
                await self._session_manager.delete_draft(context.session_id, draft.draft_id)
            except JiraIntegrationError as error:
                failed.append(
                    {
                        "draft_id": draft.draft_id,
                        "title": draft.title,
                        "error": str(error),
                    }
                )

        if context.conversation_id:
            await sync_drafts_to_conversation(context, self._session_manager)
        data = {"created": created, "failed": failed}
        message = f"Created {len(created)} Jira issue(s); {len(failed)} draft(s) failed."
        if not created:
            return ToolResult.failure(message, data=data)
        return ToolResult.success(data=data, message=message)


def _resolve_issue_type(draft_type: str, available_types: list[str]) -> str | None:
    available = {name.casefold(): name for name in available_types}
    aliases = _ISSUE_TYPE_ALIASES.get(draft_type.strip().casefold(), (draft_type,))
    return next((available[name.casefold()] for name in aliases if name.casefold() in available), None)


def _adf_text(text: str) -> dict[str, Any]:
    paragraphs = text.splitlines() or [""]
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": paragraph}],
            }
            for paragraph in paragraphs
        ],
    }


def _adf_to_plain_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""
    text = value.get("text")
    if isinstance(text, str):
        return text
    content = value.get("content")
    if not isinstance(content, list):
        return ""
    lines = [_adf_to_plain_text(item) for item in content]
    return "\n".join(line for line in lines if line)


@register_tools
def create_jira_tools(session_manager: SessionManager) -> list[BaseTool]:
    return [
        ListJiraProjectsTool(session_manager),
        SelectJiraProjectTool(session_manager),
        GetJiraIssueTypesTool(session_manager),
        SearchJiraIssuesTool(session_manager),
        GetJiraIssueTool(session_manager),
        UpdateJiraIssueTool(session_manager),
        AddJiraCommentTool(session_manager),
        PushToJiraTool(session_manager),
    ]
