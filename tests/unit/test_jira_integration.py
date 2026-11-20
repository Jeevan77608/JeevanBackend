from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from pydantic import SecretStr

from src.config.settings import Settings
from src.integrations.jira import JiraCloudClient


def test_jira_authorization_url_has_user_consent_scopes(
    mock_settings: Settings,
) -> None:
    mock_settings.jira_client_id = "atlassian-client-id"
    mock_settings.jira_client_secret = SecretStr("client-secret")
    mock_settings.jira_redirect_uri = "http://localhost:3000/api/jira/callback"
    url = JiraCloudClient(mock_settings).create_authorization_url("opaque-state")
    query = parse_qs(urlparse(url).query)

    assert urlparse(url).hostname == "auth.atlassian.com"
    assert query["client_id"] == ["atlassian-client-id"]
    assert query["state"] == ["opaque-state"]
    assert query["redirect_uri"] == [mock_settings.jira_redirect_uri]
    assert set(query["scope"][0].split()) == {
        "read:jira-work",
        "write:jira-work",
        "read:jira-user",
        "offline_access",
    }


def test_jira_tokens_are_stored_with_cloud_site_metadata() -> None:
    connection = JiraCloudClient._stored_connection(
        {
            "access_token": "private-access-token",
            "refresh_token": "private-refresh-token",
            "expires_in": 3600,
        },
        {
            "id": "cloud-id",
            "url": "https://example.atlassian.net",
            "name": "Example Jira",
        },
    )

    assert connection["cloud_id"] == "cloud-id"
    assert connection["site_url"] == "https://example.atlassian.net"
    assert connection["site_name"] == "Example Jira"
    assert connection["access_token"] == "private-access-token"
    assert connection["refresh_token"] == "private-refresh-token"
