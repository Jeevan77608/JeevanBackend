"""Per-user Jira Cloud OAuth and REST API integration."""

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx

from src.config.settings import Settings
from src.models.user import SessionContext
from src.session.manager import SessionManager

logger = logging.getLogger(__name__)


class JiraIntegrationError(Exception):
    """Raised when Jira OAuth or API operations fail."""


class JiraCloudClient:
    AUTHORIZATION_ENDPOINT = "https://auth.atlassian.com/authorize"
    TOKEN_ENDPOINT = "https://auth.atlassian.com/oauth/token"
    RESOURCES_ENDPOINT = "https://api.atlassian.com/oauth/token/accessible-resources"
    API_BASE = "https://api.atlassian.com/ex/jira"
    SCOPES = (
        "read:jira-work write:jira-work read:jira-user offline_access"
    )

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _require_configuration(self) -> None:
        if not self._settings.jira_client_id or not self._settings.jira_client_secret.get_secret_value():
            raise JiraIntegrationError("Atlassian OAuth client credentials are not configured.")
        if not self._settings.jira_redirect_uri:
            raise JiraIntegrationError("Atlassian OAuth callback URL is not configured.")

    def create_authorization_url(self, state: str) -> str:
        self._require_configuration()
        return f"{self.AUTHORIZATION_ENDPOINT}?{urlencode({
            'audience': 'api.atlassian.com',
            'client_id': self._settings.jira_client_id,
            'scope': self.SCOPES,
            'redirect_uri': self._settings.jira_redirect_uri,
            'state': state,
            'response_type': 'code',
            'prompt': 'consent',
        })}"

    async def exchange_code(self, code: str) -> dict[str, Any]:
        self._require_configuration()
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                self.TOKEN_ENDPOINT,
                json={
                    "grant_type": "authorization_code",
                    "client_id": self._settings.jira_client_id,
                    "client_secret": self._settings.jira_client_secret.get_secret_value(),
                    "code": code,
                    "redirect_uri": self._settings.jira_redirect_uri,
                },
            )
            if response.is_error:
                logger.warning("Atlassian code exchange rejected with HTTP %s", response.status_code)
                raise JiraIntegrationError("Atlassian authorization could not be completed.")
            token = response.json()

            resources_response = await client.get(
                self.RESOURCES_ENDPOINT,
                headers={"Authorization": f"Bearer {token['access_token']}"},
            )
            if resources_response.is_error:
                logger.warning(
                    "Atlassian accessible-resources request failed with HTTP %s",
                    resources_response.status_code,
                )
                raise JiraIntegrationError("Could not access the authorized Jira sites.")
            resources = resources_response.json()

        wanted_host = urlparse(self._settings.jira_site_url).hostname
        matching_resource = next(
            (
                resource
                for resource in resources
                if urlparse(resource.get("url", "")).hostname == wanted_host
            ),
            None,
        )
        if not matching_resource:
            raise JiraIntegrationError(
                "The connected Atlassian account does not have access to the configured Jira site."
            )

        return self._stored_connection(token, matching_resource)

    async def refresh_connection(self, connection: dict[str, Any]) -> dict[str, Any]:
        refresh_token = connection.get("refresh_token")
        if not isinstance(refresh_token, str) or not refresh_token:
            raise JiraIntegrationError("Jira authorization expired. Please connect Jira again.")

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                self.TOKEN_ENDPOINT,
                json={
                    "grant_type": "refresh_token",
                    "client_id": self._settings.jira_client_id,
                    "client_secret": self._settings.jira_client_secret.get_secret_value(),
                    "refresh_token": refresh_token,
                },
            )
            if response.is_error:
                logger.warning("Atlassian token refresh rejected with HTTP %s", response.status_code)
                raise JiraIntegrationError("Jira authorization expired. Please connect Jira again.")
            token = response.json()

        refreshed = dict(connection)
        refreshed.update(token)
        refreshed["refresh_token"] = token.get("refresh_token", refresh_token)
        refreshed["expires_at"] = (
            datetime.now(timezone.utc) + timedelta(seconds=int(token.get("expires_in", 3600)))
        ).isoformat()
        return refreshed

    async def request(
        self,
        session: SessionContext,
        session_manager: SessionManager,
        method: str,
        resource_path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        connection = session.integration_tokens.get("jira")
        if not connection:
            raise JiraIntegrationError("Connect your Jira account before using Jira features.")

        expires_at = connection.get("expires_at")
        if isinstance(expires_at, str):
            try:
                expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                if expiry.tzinfo is None:
                    expiry = expiry.replace(tzinfo=timezone.utc)
                if expiry <= datetime.now(timezone.utc) + timedelta(minutes=1):
                    connection = await self.refresh_connection(connection)
                    session.integration_tokens["jira"] = connection
                    await session_manager.update_session(session)
            except ValueError as error:
                raise JiraIntegrationError("Stored Jira authorization is invalid. Reconnect Jira.") from error

        url = f"{self.API_BASE}/{connection['cloud_id']}/rest/api/3/{resource_path.lstrip('/')}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method,
                url,
                headers={
                    "Authorization": f"Bearer {connection['access_token']}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                params=params,
                json=json_body,
            )

        if response.status_code == 401:
            raise JiraIntegrationError("Jira rejected the authorization. Reconnect your Jira account.")
        if response.is_error:
            logger.warning(
                "Jira API request failed with HTTP %s for %s",
                response.status_code,
                resource_path,
            )
            try:
                detail = response.json().get("errorMessages", [])
                message = "; ".join(detail) if detail else "Jira request failed."
            except (ValueError, AttributeError):
                message = "Jira request failed."
            raise JiraIntegrationError(message)
        if response.status_code == 204 or not response.content:
            return {}
        return response.json()

    @staticmethod
    def _stored_connection(
        token: dict[str, Any], resource: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "access_token": token["access_token"],
            "refresh_token": token.get("refresh_token"),
            "expires_at": (
                datetime.now(timezone.utc)
                + timedelta(seconds=int(token.get("expires_in", 3600)))
            ).isoformat(),
            "cloud_id": resource["id"],
            "site_url": resource["url"],
            "site_name": resource.get("name", ""),
        }

    @staticmethod
    def new_state() -> str:
        return secrets.token_urlsafe(32)
