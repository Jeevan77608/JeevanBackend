"""Unit tests for src/config/settings.py — Settings, sub-models, validators, and caching."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.settings import (
    AzureADSettings,
    AzureDevOpsSettings,
    AzureOpenAISettings,
    RedisSettings,
    Settings,
    clear_settings_cache,
    get_settings,
)


# ---- Helpers ----------------------------------------------------------------

_TENANT = "a" * 36
_CLIENT = "b" * 36

_MINIMAL_DEV_ENV = {
    "APP_SECRET_KEY": "test-secret-key-must-be-32-chars!",
    "AUTH_ENABLED": "false",
    "REDIS_ENABLED": "false",
    "CHAT_SAVE_ENABLED": "false",
    "APP_ENV": "development",
    "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
    "AZURE_OPENAI_API_KEY": "test-key",
}

_PROD_ENCRYPTION_KEY = "cHJvZHVjdGlvbi1rZXktZm9yLXRlc3RpbmcxMjM0"  # different from default

_PROD_ENV = {
    **_MINIMAL_DEV_ENV,
    "APP_ENV": "production",
    "APP_DEBUG": "false",
    "AUTH_ENABLED": "true",
    "AZURE_AD_TENANT_ID": _TENANT,
    "AZURE_AD_CLIENT_ID": _CLIENT,
    "AZURE_AD_CLIENT_SECRET": "super-secret",
    "AZURE_AD_REDIRECT_URI": "https://frontend.example.com/api/auth/callback",
    "CORS_ORIGINS": "https://frontend.example.com",
    "REDIS_ENABLED": "true",
    "REDIS_URL": "redis://redis.example.com:6380",
    "REDIS_SSL_ENABLED": "true",
    "ENCRYPTION_KEY": _PROD_ENCRYPTION_KEY,
    "INTERNAL_API_KEY": "prod-internal-api-key-value",
}


def _set_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    """Apply a dict of env vars via monkeypatch."""
    for key, value in env.items():
        monkeypatch.setenv(key, value)


# ---- AzureADSettings -------------------------------------------------------


class TestAzureADSettings:
    """Tests for the AzureADSettings sub-model."""

    def test_authority_property(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_AD_TENANT_ID", _TENANT)
        monkeypatch.setenv("AZURE_AD_CLIENT_ID", _CLIENT)
        monkeypatch.setenv("AZURE_AD_CLIENT_SECRET", "secret")
        monkeypatch.setenv("AZURE_AD_REDIRECT_URI", "https://example.com/callback")
        settings = AzureADSettings()
        assert settings.authority == f"https://login.microsoftonline.com/{_TENANT}"

    def test_token_endpoint_property(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_AD_TENANT_ID", _TENANT)
        monkeypatch.setenv("AZURE_AD_CLIENT_ID", _CLIENT)
        monkeypatch.setenv("AZURE_AD_CLIENT_SECRET", "secret")
        monkeypatch.setenv("AZURE_AD_REDIRECT_URI", "https://example.com/callback")
        settings = AzureADSettings()
        expected = f"https://login.microsoftonline.com/{_TENANT}/oauth2/v2.0/token"
        assert settings.token_endpoint == expected

    def test_authorize_endpoint_property(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_AD_TENANT_ID", _TENANT)
        monkeypatch.setenv("AZURE_AD_CLIENT_ID", _CLIENT)
        monkeypatch.setenv("AZURE_AD_CLIENT_SECRET", "secret")
        monkeypatch.setenv("AZURE_AD_REDIRECT_URI", "https://example.com/callback")
        settings = AzureADSettings()
        expected = f"https://login.microsoftonline.com/{_TENANT}/oauth2/v2.0/authorize"
        assert settings.authorize_endpoint == expected

    def test_scopes_property(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_AD_TENANT_ID", _TENANT)
        monkeypatch.setenv("AZURE_AD_CLIENT_ID", _CLIENT)
        monkeypatch.setenv("AZURE_AD_CLIENT_SECRET", "secret")
        monkeypatch.setenv("AZURE_AD_REDIRECT_URI", "https://example.com/callback")
        settings = AzureADSettings()
        scopes = settings.scopes
        assert "openid" in scopes
        assert "profile" in scopes
        assert "offline_access" in scopes
        assert "499b84ac-1321-427f-aa17-267ca6975798/.default" in scopes
        assert len(scopes) == 4


# ---- AzureOpenAISettings ---------------------------------------------------


class TestAzureOpenAISettings:
    """Tests for the AzureOpenAISettings sub-model."""

    def test_validate_endpoint_adds_trailing_slash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
        settings = AzureOpenAISettings()
        assert settings.endpoint == "https://test.openai.azure.com/"

    def test_validate_endpoint_preserves_existing_slash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
        settings = AzureOpenAISettings()
        assert settings.endpoint == "https://test.openai.azure.com/"

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        monkeypatch.setenv("AZURE_OPENAI_MAX_TOKENS", "4000")
        settings = AzureOpenAISettings()
        assert settings.deployment_name == "gpt-4"
        assert settings.api_version == "2024-02-01"
        assert settings.max_tokens == 4000
        assert settings.temperature == pytest.approx(0.7)


# ---- RedisSettings ----------------------------------------------------------


class TestRedisSettings:
    """Tests for the RedisSettings sub-model."""

    def test_connection_url_ssl_enabled_rewrites_scheme(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REDIS_URL", "redis://redis.example.com:6380")
        monkeypatch.setenv("REDIS_SSL_ENABLED", "true")
        settings = RedisSettings()
        assert settings.connection_url == "rediss://redis.example.com:6380"

    def test_connection_url_ssl_disabled_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REDIS_URL", "redis://redis.example.com:6379")
        monkeypatch.setenv("REDIS_SSL_ENABLED", "false")
        settings = RedisSettings()
        assert settings.connection_url == "redis://redis.example.com:6379"

    def test_connection_url_already_rediss_with_ssl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the URL already starts with rediss://, ssl_enabled=True should not double-replace."""
        monkeypatch.setenv("REDIS_URL", "rediss://redis.example.com:6380")
        monkeypatch.setenv("REDIS_SSL_ENABLED", "true")
        settings = RedisSettings()
        # Does not start with "redis://" so replace is a no-op, returns as-is
        assert settings.connection_url == "rediss://redis.example.com:6380"


# ---- AzureDevOpsSettings ---------------------------------------------------


class TestAzureDevOpsSettings:
    """Tests for the AzureDevOpsSettings sub-model."""

    def test_validate_org_url_strips_trailing_slash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AZURE_DEVOPS_DEFAULT_ORG_URL", "https://dev.azure.com/myorg/")
        settings = AzureDevOpsSettings()
        assert settings.default_org_url == "https://dev.azure.com/myorg"

    def test_validate_org_url_none_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # default_org_url defaults to None when not set via env
        monkeypatch.delenv("AZURE_DEVOPS_DEFAULT_ORG_URL", raising=False)
        settings = AzureDevOpsSettings(_env_file=None)
        assert settings.default_org_url is None


# ---- Settings: environment helpers -----------------------------------------


class TestSettingsEnvironment:
    """Tests for is_development / is_production properties."""

    def test_is_development(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_env(monkeypatch, _MINIMAL_DEV_ENV)
        clear_settings_cache()
        settings = Settings()
        assert settings.is_development is True
        assert settings.is_production is False
        clear_settings_cache()

    def test_is_production(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_env(monkeypatch, _PROD_ENV)
        clear_settings_cache()
        settings = Settings()
        assert settings.is_production is True
        assert settings.is_development is False
        clear_settings_cache()


# ---- Settings: CORS parsing ------------------------------------------------


class TestCorsOrigins:
    """Tests for parse_cors_origins validator and get_cors_origins_list."""

    def test_parse_cors_origins_none_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_env(monkeypatch, _MINIMAL_DEV_ENV)
        monkeypatch.delenv("CORS_ORIGINS", raising=False)
        clear_settings_cache()
        settings = Settings()
        assert settings.cors_origins == ""
        clear_settings_cache()

    def test_parse_cors_origins_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_env(monkeypatch, _MINIMAL_DEV_ENV)
        monkeypatch.setenv("CORS_ORIGINS", "")
        clear_settings_cache()
        settings = Settings()
        assert settings.cors_origins == ""
        clear_settings_cache()

    def test_parse_cors_origins_comma_separated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_env(monkeypatch, _MINIMAL_DEV_ENV)
        monkeypatch.setenv("CORS_ORIGINS", "https://a.com,https://b.com")
        clear_settings_cache()
        settings = Settings()
        assert settings.cors_origins == "https://a.com,https://b.com"
        clear_settings_cache()

    def test_parse_cors_origins_json_array(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_env(monkeypatch, _MINIMAL_DEV_ENV)
        monkeypatch.setenv("CORS_ORIGINS", '["https://a.com", "https://b.com"]')
        clear_settings_cache()
        settings = Settings()
        assert settings.cors_origins == "https://a.com,https://b.com"
        clear_settings_cache()

    def test_parse_cors_origins_list_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When parse_cors_origins receives a Python list directly (e.g. from code), it joins."""
        result = Settings.parse_cors_origins(["https://a.com", "https://b.com"])
        assert result == "https://a.com,https://b.com"

    def test_get_cors_origins_list_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_env(monkeypatch, _MINIMAL_DEV_ENV)
        monkeypatch.setenv("CORS_ORIGINS", "")
        clear_settings_cache()
        settings = Settings()
        assert settings.get_cors_origins_list() == []
        clear_settings_cache()

    def test_get_cors_origins_list_splits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_env(monkeypatch, _MINIMAL_DEV_ENV)
        monkeypatch.setenv("CORS_ORIGINS", "https://a.com,https://b.com")
        clear_settings_cache()
        settings = Settings()
        assert settings.get_cors_origins_list() == ["https://a.com", "https://b.com"]
        clear_settings_cache()


# ---- Settings: validate_settings (model_validator) -------------------------


class TestValidateSettings:
    """Tests for the model_validator that enforces cross-field invariants."""

    def test_auth_enabled_without_azure_ad_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_env(monkeypatch, _MINIMAL_DEV_ENV)
        monkeypatch.setenv("AUTH_ENABLED", "true")
        # Override AZURE_AD_* vars (delenv alone won't work because .env still provides them)
        monkeypatch.setenv("AZURE_AD_TENANT_ID", "")
        monkeypatch.setenv("AZURE_AD_CLIENT_ID", "")
        monkeypatch.setenv("AZURE_AD_CLIENT_SECRET", "")
        monkeypatch.setenv("AZURE_AD_REDIRECT_URI", "")
        clear_settings_cache()
        with pytest.raises(ValidationError, match="Azure AD settings required"):
            Settings()
        clear_settings_cache()

    def test_google_auth_does_not_require_azure_ad(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_env(monkeypatch, _MINIMAL_DEV_ENV)
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("AUTH_PROVIDER", "google")
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "google-client-id")
        monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "google-client-secret")
        monkeypatch.setenv(
            "GOOGLE_REDIRECT_URI", "http://localhost:3000/api/auth/callback"
        )
        monkeypatch.setenv("AZURE_AD_TENANT_ID", "")
        monkeypatch.setenv("AZURE_AD_CLIENT_ID", "")
        monkeypatch.setenv("AZURE_AD_CLIENT_SECRET", "")
        monkeypatch.setenv("AZURE_AD_REDIRECT_URI", "")
        clear_settings_cache()

        settings = Settings()

        assert settings.auth_provider == "google"
        assert settings.google_client_id == "google-client-id"
        clear_settings_cache()

    def test_auth_enabled_with_pat_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_env(monkeypatch, _MINIMAL_DEV_ENV)
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("AZURE_AD_TENANT_ID", _TENANT)
        monkeypatch.setenv("AZURE_AD_CLIENT_ID", _CLIENT)
        monkeypatch.setenv("AZURE_AD_CLIENT_SECRET", "secret")
        monkeypatch.setenv("AZURE_AD_REDIRECT_URI", "https://example.com/callback")
        monkeypatch.setenv("AZURE_DEVOPS_PAT", "my-pat-token")
        clear_settings_cache()
        with pytest.raises(ValidationError, match="PAT.*not allowed when AUTH_ENABLED=true"):
            Settings()
        clear_settings_cache()

    def test_production_debug_true_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_env(monkeypatch, _PROD_ENV)
        monkeypatch.setenv("APP_DEBUG", "true")
        clear_settings_cache()
        with pytest.raises(ValidationError, match="Debug mode must be disabled in production"):
            Settings()
        clear_settings_cache()

    def test_production_auth_disabled_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_env(monkeypatch, _PROD_ENV)
        monkeypatch.setenv("AUTH_ENABLED", "false")
        clear_settings_cache()
        with pytest.raises(ValidationError, match="Authentication must be enabled in production"):
            Settings()
        clear_settings_cache()

    def test_production_no_cors_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_env(monkeypatch, _PROD_ENV)
        monkeypatch.setenv("CORS_ORIGINS", "")
        clear_settings_cache()
        with pytest.raises(ValidationError, match="CORS_ORIGINS.*required in production"):
            Settings()
        clear_settings_cache()

    def test_production_redis_enabled_no_url_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_env(monkeypatch, _PROD_ENV)
        monkeypatch.setenv("REDIS_ENABLED", "true")
        monkeypatch.setenv("REDIS_URL", "")
        clear_settings_cache()
        with pytest.raises(ValidationError, match="REDIS_URL.*required"):
            Settings()
        clear_settings_cache()

    def test_production_redis_no_tls_non_localhost_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_env(monkeypatch, _PROD_ENV)
        monkeypatch.setenv("REDIS_SSL_ENABLED", "false")
        clear_settings_cache()
        with pytest.raises(ValidationError, match="Redis TLS should be enabled in production"):
            Settings()
        clear_settings_cache()

    def test_production_default_encryption_key_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_env(monkeypatch, _PROD_ENV)
        monkeypatch.setenv("ENCRYPTION_KEY", "ZGV2ZWxvcG1lbnQta2V5LWZvci10ZXN0aW5nMTIz")
        clear_settings_cache()
        with pytest.raises(
            ValidationError, match="Default development encryption key must not be used"
        ):
            Settings()
        clear_settings_cache()

    def test_production_no_internal_api_key_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_env(monkeypatch, _PROD_ENV)
        monkeypatch.setenv("INTERNAL_API_KEY", "")
        clear_settings_cache()
        with pytest.raises(ValidationError, match="INTERNAL_API_KEY is required in production"):
            Settings()
        clear_settings_cache()

    def test_chat_save_enabled_no_cosmos_endpoint_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_env(monkeypatch, _MINIMAL_DEV_ENV)
        monkeypatch.setenv("CHAT_SAVE_ENABLED", "true")
        monkeypatch.setenv("COSMOS_ENDPOINT", "")
        monkeypatch.setenv("COSMOS_KEY", "some-key")
        clear_settings_cache()
        with pytest.raises(
            ValidationError, match="COSMOS_ENDPOINT and COSMOS_KEY are required"
        ):
            Settings()
        clear_settings_cache()

    def test_chat_save_enabled_no_cosmos_key_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_env(monkeypatch, _MINIMAL_DEV_ENV)
        monkeypatch.setenv("CHAT_SAVE_ENABLED", "true")
        monkeypatch.setenv("COSMOS_ENDPOINT", "https://my-cosmos.documents.azure.com:443/")
        monkeypatch.setenv("COSMOS_KEY", "")
        clear_settings_cache()
        with pytest.raises(
            ValidationError, match="COSMOS_ENDPOINT and COSMOS_KEY are required"
        ):
            Settings()
        clear_settings_cache()

    def test_chat_save_enabled_localhost_cosmos_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_env(monkeypatch, _MINIMAL_DEV_ENV)
        monkeypatch.setenv("CHAT_SAVE_ENABLED", "true")
        monkeypatch.setenv("COSMOS_ENDPOINT", "https://localhost:8081/")
        monkeypatch.setenv("COSMOS_KEY", "some-key")
        clear_settings_cache()
        with pytest.raises(ValidationError, match="localhost.*not supported"):
            Settings()
        clear_settings_cache()

    def test_valid_production_settings_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A fully valid production configuration should not raise."""
        _set_env(monkeypatch, _PROD_ENV)
        clear_settings_cache()
        settings = Settings()
        assert settings.is_production is True
        assert settings.auth_enabled is True
        clear_settings_cache()

    def test_valid_dev_settings_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Minimal dev configuration should construct without errors."""
        _set_env(monkeypatch, _MINIMAL_DEV_ENV)
        clear_settings_cache()
        settings = Settings()
        assert settings.is_development is True
        assert settings.auth_enabled is False
        clear_settings_cache()


# ---- Settings: get_settings / clear_settings_cache caching -----------------


class TestSettingsCache:
    """Tests for LRU caching of get_settings and clear_settings_cache."""

    def test_get_settings_returns_same_instance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_env(monkeypatch, _MINIMAL_DEV_ENV)
        clear_settings_cache()
        first = get_settings()
        second = get_settings()
        assert first is second
        clear_settings_cache()

    def test_clear_settings_cache_forces_new_instance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_env(monkeypatch, _MINIMAL_DEV_ENV)
        clear_settings_cache()
        first = get_settings()
        clear_settings_cache()
        second = get_settings()
        assert first is not second
        clear_settings_cache()
