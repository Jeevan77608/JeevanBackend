"""
Application settings management using Pydantic Settings.

This module follows the 12-Factor App methodology for configuration:
- Configuration is stored in environment variables
- Sensitive values use SecretStr for secure handling
- Settings are validated at startup
- Hierarchical configuration with sensible defaults

References:
- https://12factor.net/config
- https://pydantic-docs.helpmanual.io/usage/settings/
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import (
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# =============================================================================
# Azure AD / Entra ID Configuration
# =============================================================================


class AzureADSettings(BaseSettings):
    """
    Azure AD / Entra ID OAuth 2.0 configuration.

    Supports both Azure DevOps Services (cloud) and Azure DevOps Server (on-prem).
    Uses delegated permissions with OAuth 2.0 Authorization Code + PKCE flow.
    """

    model_config = SettingsConfigDict(
        # Use absolute path to .env file to ensure it's found regardless of working directory
        env_file=str(Path(__file__).parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        env_prefix="AZURE_AD_",
        extra="ignore",
    )

    tenant_id: str = Field(
        ...,
        min_length=36,
        max_length=36,
        description="Azure AD Tenant ID (GUID format)",
    )
    client_id: str = Field(
        ...,
        min_length=36,
        max_length=36,
        description="Azure AD Application (Client) ID",
    )
    client_secret: SecretStr = Field(
        ...,
        min_length=1,
        description="Azure AD Client Secret",
    )
    redirect_uri: str = Field(
        ...,
        description="OAuth 2.0 redirect URI (must match Azure AD app registration)",
    )

    @property
    def authority(self) -> str:
        """Azure AD authority URL for authentication."""
        return f"https://login.microsoftonline.com/{self.tenant_id}"

    @property
    def token_endpoint(self) -> str:
        """OAuth 2.0 token endpoint."""
        return f"{self.authority}/oauth2/v2.0/token"

    @property
    def authorize_endpoint(self) -> str:
        """OAuth 2.0 authorization endpoint."""
        return f"{self.authority}/oauth2/v2.0/authorize"

    @property
    def scopes(self) -> list[str]:
        """
        Required OAuth scopes for Azure DevOps access.

        We explicitly request openid and profile scopes to get ID token with user info.
        The .default scope provides Azure DevOps API access.

        Note: openid and profile are standard OAuth scopes (not resource-specific),
        so they can be combined with .default scope without issues.
        """
        return [
            "openid",  # Required for ID token
            "profile",  # Required for user profile claims (name, email, etc.)
            "offline_access",  # Required for refresh token
            "499b84ac-1321-427f-aa17-267ca6975798/.default",  # Azure DevOps API access
            # openid + profile + offline_access give us ID token with user info
            # .default gives us Azure DevOps API access
        ]


# =============================================================================
# Azure OpenAI Configuration
# =============================================================================


class AzureOpenAISettings(BaseSettings):
    """
    Azure OpenAI service configuration.

    Supports GPT-4 and other models deployed in Azure OpenAI.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AZURE_OPENAI_",
        extra="ignore",
    )

    endpoint: str = Field(
        ...,
        description="Azure OpenAI endpoint URL (e.g., https://your-resource.openai.azure.com/)",
    )
    api_key: SecretStr = Field(
        ...,
        min_length=1,
        description="Azure OpenAI API key",
    )
    deployment_name: str = Field(
        default="gpt-4",
        description="Model deployment name in Azure OpenAI",
    )
    api_version: str = Field(
        default="2024-02-01",
        description="Azure OpenAI API version",
    )

    # Generation settings
    max_tokens: int = Field(
        default=4000,
        ge=100,
        le=128000,
        description="Maximum tokens for generation",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Generation temperature (0=deterministic, 2=creative)",
    )

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        """Ensure endpoint ends with /."""
        return v.rstrip("/") + "/"


# =============================================================================
# Redis Configuration
# =============================================================================


class RedisSettings(BaseSettings):
    """
    Redis connection and pooling configuration.

    Supports both Redis standalone and Azure Cache for Redis.
    Includes connection pooling for production workloads.
    """

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        env_prefix="REDIS_",
        extra="ignore",
    )

    url: str = Field(
        default="",
        description="Redis connection URL (redis:// or rediss:// for TLS). Required when REDIS_ENABLED=true.",
    )
    password: SecretStr | None = Field(
        default=None,
        description="Redis password (if not in URL)",
    )

    # Connection pool settings
    max_connections: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum connections in pool",
    )
    socket_timeout: float = Field(
        default=5.0,
        ge=0.5,
        le=30.0,
        description="Socket timeout in seconds",
    )
    connect_timeout: float = Field(
        default=5.0,
        ge=0.5,
        le=30.0,
        description="Connection timeout in seconds",
    )

    # TLS settings (for Azure Cache for Redis)
    ssl_enabled: bool = Field(
        default=False,
        description="Enable TLS/SSL connection",
    )

    @property
    def connection_url(self) -> str:
        """Get connection URL with TLS if enabled."""
        if self.ssl_enabled and self.url.startswith("redis://"):
            return self.url.replace("redis://", "rediss://", 1)
        return self.url


# =============================================================================
# Azure Cosmos DB Configuration (conversation history)
# =============================================================================


class CosmosSettings(BaseSettings):
    """
    Azure Cosmos DB configuration for chat conversation persistence.

    Used only when CHAT_SAVE_ENABLED=true. Partition key must be user_id.
    Document TTL is set per-document from CHAT_RETENTION_DAYS.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="COSMOS_",
        extra="ignore",
    )

    endpoint: str = Field(
        default="",
        description="Cosmos DB endpoint URL (e.g. https://your-account.documents.azure.com:443/)",
    )
    key: SecretStr = Field(
        default=SecretStr(""),
        description="Cosmos DB read-write key",
    )
    database_id: str = Field(
        default="mcp",
        description="Database name",
    )
    container_id: str = Field(
        default="conversations",
        description="Container name (partition key: /user_id)",
    )
    serverless: bool = Field(
        default=False,
        description="True if using a serverless Cosmos DB account (do not set offer throughput on container)",
    )


# =============================================================================
# Azure DevOps Configuration
# =============================================================================


class AzureDevOpsSettings(BaseSettings):
    """
    Azure DevOps API configuration.

    Supports both Azure DevOps Services (cloud) and Azure DevOps Server (on-prem).
    Uses REST API v7.0 (LTS version).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AZURE_DEVOPS_",
        extra="ignore",
    )

    default_org_url: str | None = Field(
        default=None,
        description="Default organization URL (e.g., https://dev.azure.com/myorg)",
    )
    default_project: str | None = Field(
        default=None,
        description="Default project name",
    )
    wiki_identifier: str | None = Field(
        default=None,
        description="Wiki identifier (e.g., ProjectName.wiki)",
    )
    pat: SecretStr | None = Field(
        default=None,
        description="Personal Access Token (used when AUTH_ENABLED=false)",
    )
    api_version: str = Field(
        default="7.0",
        description="Azure DevOps REST API version",
    )

    # Rate limiting
    requests_per_second: float = Field(
        default=10.0,
        ge=1.0,
        le=100.0,
        description="Maximum API requests per second",
    )
    retry_max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for failed requests",
    )

    # Timeouts
    request_timeout: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="HTTP request timeout in seconds",
    )

    @field_validator("default_org_url")
    @classmethod
    def validate_org_url(cls, v: str | None) -> str | None:
        """Normalize organization URL."""
        if v:
            return v.rstrip("/")
        return v


# =============================================================================
# Helper Functions
# =============================================================================


def _load_azure_ad_settings() -> AzureADSettings | None:
    """
    Conditionally load Azure AD settings from environment variables.

    Returns AzureADSettings instance if all required env vars are present,
    None otherwise.

    Note: Pydantic Settings automatically loads from .env file when env_file is specified.
    The .env file path is resolved relative to the current working directory.
    """
    try:
        # Try to create AzureADSettings - it will read from .env file automatically
        # If required fields are missing, ValidationError will be raised
        # We catch all exceptions and return None, letting validate_settings() provide a better error
        settings = AzureADSettings()
        # Verify that required fields are actually set (not empty strings)
        if settings.tenant_id and settings.client_id and settings.client_secret.get_secret_value():
            return settings
        return None
    except Exception:
        # If loading fails (e.g., missing required fields, validation error), return None
        # The validation in validate_settings will catch this and provide a better error
        return None


# =============================================================================
# Main Application Settings
# =============================================================================


class Settings(BaseSettings):
    """
    Main application settings aggregating all configuration.

    Environment variables are loaded from .env file in development.
    All sensitive values use SecretStr for secure handling.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
        env_prefix="",  # No prefix for top-level settings
    )

    # Application metadata
    app_name: str = Field(
        default="Jeevan AI Chatbot",
        description="Application name",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application version",
    )
    app_env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment",
    )
    auth_provider: Literal["azure_ad", "google", "supabase"] = Field(
        default="azure_ad",
        description="Identity provider used for interactive sign-in",
    )
    google_client_id: str = Field(default="", description="Google OAuth client ID")
    google_client_secret: SecretStr = Field(
        default=SecretStr(""), description="Google OAuth client secret"
    )
    google_redirect_uri: str = Field(
        default="http://localhost:3000/api/auth/callback",
        description="Registered Google OAuth callback URL",
    )
    supabase_url: str = Field(default="", description="Supabase project URL")
    supabase_anon_key: SecretStr = Field(
        default=SecretStr(""), description="Supabase public anonymous key"
    )
    supabase_service_role_key: SecretStr = Field(
        default=SecretStr(""), description="Supabase server-side service role key"
    )
    supabase_storage_bucket: str = Field(
        default="user-context-files",
        description="Private Supabase Storage bucket for extracted file context",
    )
    supabase_storage_enabled: bool = Field(
        default=False,
        description="Store uploaded file context in private Supabase Storage",
    )
    jira_client_id: str = Field(default="", description="Atlassian OAuth client ID")
    jira_client_secret: SecretStr = Field(
        default=SecretStr(""), description="Atlassian OAuth client secret"
    )
    jira_redirect_uri: str = Field(
        default="http://localhost:3000/api/jira/callback",
        description="Registered Atlassian OAuth callback URL",
    )
    jira_site_url: str = Field(
        default="https://your-site.atlassian.net",
        description="Organization Jira Cloud site URL",
    )
    app_debug: bool = Field(
        default=False,
        description="Enable debug mode (additional logging, docs endpoints)",
    )
    app_secret_key: SecretStr = Field(
        ...,
        min_length=32,
        description="Application secret key for signing (min 32 chars)",
    )

    # Feature toggles
    auth_enabled: bool = Field(
        default=True,
        description="Enable/disable authentication (disable for local development)",
    )
    redis_enabled: bool = Field(
        default=True,
        description="Enable/disable Redis (uses in-memory storage when disabled)",
    )
    chat_save_enabled: bool = Field(
        default=False,
        description="Enable/disable saving chat conversations to Cosmos DB (hamburger history)",
    )
    chat_retention_days: int = Field(
        default=15,
        ge=1,
        le=365,
        description="Number of days to retain saved conversations (Cosmos TTL); configurable",
    )
    chat_history_display_days: int = Field(
        default=15,
        ge=1,
        le=365,
        description="Number of days of conversation history to display in sidebar (env: CHAT_HISTORY_DISPLAY_DAYS)",
    )

    # Chat scope and context (epic/task generation only; limit context to reduce hallucination)
    chat_scope_strict: bool = Field(
        default=True,
        description="Restrict conversation to Epic and Task generation and related setup only; prompt boundary message otherwise",
    )
    chat_scope_boundary_message: str = Field(
        default="This chat is for Epic and Task generation with Azure DevOps only. Please ask only about connecting to your organization, selecting a project, generating epics or tasks, listing or pushing drafts, or related setup.",
        description="Message shown when user message is out of scope",
    )
    chat_context_window_messages: int = Field(
        default=20,
        ge=2,
        le=100,
        description="Max number of messages (including system) sent to the LLM; configurable context window",
    )
    chat_history_limit_messages: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Max number of history messages to include (user + assistant) in each request",
    )

    # Push settings
    default_push_tag: str = Field(
        default="@AIGenerated",
        description="Default tag injected on every push to Azure DevOps (env: DEFAULT_PUSH_TAG)",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_format: Literal["json", "text"] = Field(
        default="text",
        description="Log output format",
    )

    # Security
    encryption_key: SecretStr = Field(
        default=SecretStr("ZGV2ZWxvcG1lbnQta2V5LWZvci10ZXN0aW5nMTIz"),
        description="AES-256 encryption key (base64 encoded 32 bytes)",
    )
    internal_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Shared secret between frontend and backend for server-to-server auth (env: INTERNAL_API_KEY)",
    )

    # Session settings
    session_expiry_hours: int = Field(
        default=24,
        ge=1,
        le=168,  # Max 1 week
        description="Session idle timeout in hours",
    )

    # CORS settings (for API access)
    # Accepts comma-separated string from env var: CORS_ORIGINS=https://frontend.com,https://other.com
    # Store as string to avoid Pydantic trying to JSON decode list fields
    cors_origins: str = Field(
        default="",
        description="CORS origins as comma-separated string from CORS_ORIGINS env var",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str] | None) -> str:
        """Parse CORS_ORIGINS from env var - handle string, list, or None."""
        if v is None:
            return ""
        if isinstance(v, list):
            # If already a list (from JSON), join with comma
            return ",".join(str(item).strip() for item in v if item)
        # Handle string - could be empty, comma-separated, or JSON string
        v_str = str(v).strip()
        if not v_str:
            return ""
        # If it looks like JSON array, try to parse it
        if v_str.startswith("[") and v_str.endswith("]"):
            try:
                parsed = json.loads(v_str)
                if isinstance(parsed, list):
                    return ",".join(str(item).strip() for item in parsed if item)
            except (json.JSONDecodeError, ValueError):
                pass
        # Otherwise treat as comma-separated string
        return v_str

    def get_cors_origins_list(self) -> list[str]:
        """Get CORS origins as a list."""
        if not self.cors_origins:
            return []
        # Split by comma and strip whitespace, filter empty strings
        origins = [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
        return origins

    # Sub-configurations (loaded from prefixed env vars)
    azure_ad: AzureADSettings | None = Field(default_factory=lambda: _load_azure_ad_settings())
    azure_openai: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    cosmos: CosmosSettings = Field(default_factory=CosmosSettings)
    azure_devops: AzureDevOpsSettings = Field(default_factory=AzureDevOpsSettings)

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.app_env == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.app_env == "production"

    @model_validator(mode="after")
    def validate_settings(self) -> Settings:
        """Validate settings based on feature flags and environment."""
        # Auth enabled requires Azure AD settings
        if self.auth_enabled and self.auth_provider == "azure_ad" and self.azure_ad is None:
            raise ValueError(
                "Azure AD settings required when authentication is enabled. "
                "Set AUTH_PROVIDER=google or provide AZURE_AD_* environment variables."
            )
        if self.auth_enabled and self.auth_provider == "google":
            missing_google_settings = [
                name
                for name, value in (
                    ("GOOGLE_CLIENT_ID", self.google_client_id),
                    ("GOOGLE_CLIENT_SECRET", self.google_client_secret.get_secret_value()),
                    ("GOOGLE_REDIRECT_URI", self.google_redirect_uri),
                )
                if not value
            ]
            if missing_google_settings:
                raise ValueError(
                    "Google sign-in requires: " + ", ".join(missing_google_settings)
                )
            google_redirect = urlparse(self.google_redirect_uri)
            if google_redirect.scheme not in {"http", "https"} or not google_redirect.hostname:
                raise ValueError("GOOGLE_REDIRECT_URI must be an absolute HTTP(S) URL.")
            if self.is_production and google_redirect.scheme != "https":
                raise ValueError("GOOGLE_REDIRECT_URI must use HTTPS in production.")
        if self.auth_enabled and self.auth_provider == "supabase":
            missing_supabase_settings = [
                name
                for name, value in (
                    ("SUPABASE_URL", self.supabase_url),
                    ("SUPABASE_ANON_KEY", self.supabase_anon_key.get_secret_value()),
                    (
                        "SUPABASE_SERVICE_ROLE_KEY",
                        self.supabase_service_role_key.get_secret_value(),
                    ),
                )
                if not value
            ]
            if missing_supabase_settings:
                raise ValueError(
                    "Supabase sign-in requires: " + ", ".join(missing_supabase_settings)
                )
            supabase_url = urlparse(self.supabase_url)
            if supabase_url.scheme not in {"http", "https"} or not supabase_url.hostname:
                raise ValueError("SUPABASE_URL must be an absolute HTTP(S) URL.")
            if self.is_production and supabase_url.scheme != "https":
                raise ValueError("SUPABASE_URL must use HTTPS in production.")
        if self.supabase_storage_enabled and (
            not self.supabase_url
            or not self.supabase_service_role_key.get_secret_value()
        ):
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required when "
                "SUPABASE_STORAGE_ENABLED=true."
            )

        jira_site = urlparse(self.jira_site_url)
        if jira_site.scheme != "https" or not jira_site.hostname:
            raise ValueError("JIRA_SITE_URL must be an absolute HTTPS URL.")

        # STRICT: When AUTH_ENABLED=true, PAT must NOT be used
        if (
            self.auth_enabled
            and self.auth_provider == "azure_ad"
            and self.azure_devops.pat
        ):
            raise ValueError(
                "PAT (Personal Access Token) is not allowed when AUTH_ENABLED=true. "
                "Azure AD OAuth flow must be used exclusively. "
                "Remove AZURE_DEVOPS_PAT from environment variables or set AUTH_ENABLED=false."
            )

        # Production-specific requirements
        if self.is_production:
            # Ensure debug is disabled in production
            if self.app_debug:
                raise ValueError("Debug mode must be disabled in production")

            # Auth must be enabled in production
            if not self.auth_enabled:
                raise ValueError("Authentication must be enabled in production")

            # CORS_ORIGINS must be set in production
            cors_origins_list = self.get_cors_origins_list()
            if not cors_origins_list:
                raise ValueError(
                    "CORS_ORIGINS environment variable is required in production. "
                    "Set CORS_ORIGINS to a comma-separated list of allowed frontend URLs."
                )

            # Redis URL must be set if Redis is enabled
            if self.redis_enabled and not self.redis.url:
                raise ValueError(
                    "REDIS_URL environment variable is required when REDIS_ENABLED=true. "
                    "Set REDIS_URL or set REDIS_ENABLED=false to use in-memory storage."
                )

            # Ensure Redis TLS in production (only if Redis is enabled)
            if self.redis_enabled and not self.redis.ssl_enabled and "localhost" not in self.redis.url:
                raise ValueError("Redis TLS should be enabled in production when Redis is enabled")

            # Reject default/development encryption key in production
            _default_dev_key = "ZGV2ZWxvcG1lbnQta2V5LWZvci10ZXN0aW5nMTIz"
            if self.encryption_key.get_secret_value() == _default_dev_key:
                raise ValueError(
                    "Default development encryption key must not be used in production. "
                    "Set ENCRYPTION_KEY to a unique base64-encoded 32-byte key."
                )

            # Require INTERNAL_API_KEY in production for server-to-server auth
            if not self.internal_api_key.get_secret_value():
                raise ValueError(
                    "INTERNAL_API_KEY is required in production for server-to-server authentication. "
                    "Set INTERNAL_API_KEY to a secure random string shared with the frontend."
                )

        if self.chat_save_enabled:
            if self.auth_provider == "supabase":
                if (
                    not self.supabase_url
                    or not self.supabase_service_role_key.get_secret_value()
                ):
                    raise ValueError(
                        "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required "
                        "when CHAT_SAVE_ENABLED=true with Supabase Auth."
                    )
            elif not self.cosmos.endpoint or not self.cosmos.key.get_secret_value():
                raise ValueError(
                    "COSMOS_ENDPOINT and COSMOS_KEY are required when "
                    "CHAT_SAVE_ENABLED=true with a non-Supabase auth provider."
                )

        return self


# =============================================================================
# Settings Access
# =============================================================================


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    Settings are loaded once and cached for the lifetime of the application.
    To reload settings, call get_settings.cache_clear() first.

    Returns:
        Application settings instance
    """
    return Settings()


def clear_settings_cache() -> None:
    """Clear the settings cache (useful for testing)."""
    get_settings.cache_clear()
