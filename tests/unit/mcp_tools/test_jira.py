from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_tools.base import ToolContext, ToolStatus
from src.mcp_tools.jira import (
    ListJiraProjectsTool,
    SearchJiraIssuesTool,
    _adf_to_plain_text,
    _adf_text,
    _resolve_issue_type,
)


def test_resolve_issue_type_uses_project_available_types() -> None:
    assert _resolve_issue_type("User Story", ["Bug", "Story", "Task"]) == "Story"
    assert _resolve_issue_type("Epic", ["Task"]) is None
    assert _resolve_issue_type("Bug", ["Story", "Task"]) is None


def test_adf_text_builds_paragraphs() -> None:
    assert _adf_text("First line\nSecond line") == {
        "type": "doc",
        "version": 1,
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "First line"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": "Second line"}]},
        ],
    }


def test_adf_to_plain_text_extracts_issue_description() -> None:
    assert _adf_to_plain_text({
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "Login"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": "Requirements"}]},
        ],
    }) == "Login\nRequirements"


@pytest.mark.asyncio
async def test_list_jira_projects_uses_session_connection_without_app_oauth_token() -> None:
    session = SimpleNamespace(
        integration_tokens={
            "jira": {
                "site_url": "https://example.atlassian.net",
                "cloud_id": "jira-cloud-id",
            }
        }
    )
    manager = MagicMock()
    manager.get_session = AsyncMock(return_value=session)
    tool = ListJiraProjectsTool(manager)
    tool._client.request = AsyncMock(
        return_value={
            "values": [
                {"key": "MDP", "name": "My discovery project"},
                {"key": "WAD", "name": "Web Application Development"},
            ]
        }
    )

    result = await tool.execute(
        ToolContext(
            session_id="session-id",
            user_id="supabase:user",
        )
    )

    assert result.status == ToolStatus.SUCCESS
    assert [project["key"] for project in result.data["projects"]] == ["MDP", "WAD"]
    tool._client.request.assert_awaited_once_with(
        session,
        manager,
        "GET",
        "project/search",
        params={"maxResults": 100, "orderBy": "name"},
        json_body=None,
    )


@pytest.mark.asyncio
async def test_search_jira_issues_scopes_query_to_selected_project() -> None:
    session = SimpleNamespace(
        integration_tokens={"jira": {"site_url": "https://example.atlassian.net"}}
    )
    manager = MagicMock()
    manager.get_session = AsyncMock(return_value=session)
    tool = SearchJiraIssuesTool(manager)
    tool._request = AsyncMock(
        return_value={
            "issues": [
                {
                    "key": "MDP-12",
                    "fields": {
                        "summary": "Login issue",
                        "description": {
                            "type": "doc",
                            "content": [
                                {"type": "paragraph", "content": [{"type": "text", "text": "Login requirements"}]}
                            ],
                        },
                        "status": {"name": "Open"},
                        "issuetype": {"name": "Bug"},
                    },
                }
            ]
        }
    )

    result = await tool.execute(
        ToolContext(
            session_id="session-id",
            user_id="google:user",
            project_id="MDP",
            access_token="google-access-token",
        ),
        issue_key="MDP-12",
        issue_type="Bug",
        summary_contains='login "flow"',
    )

    assert result.status == ToolStatus.SUCCESS
    assert result.data["issues"][0]["url"] == "https://example.atlassian.net/browse/MDP-12"
    assert result.data["issues"][0]["description"] == "Login requirements"
    assert result.data["jql"] == 'project = MDP AND key = "MDP-12" AND issuetype = "Bug" AND summary ~ "login \\"flow\\"" ORDER BY updated DESC'
