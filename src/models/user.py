"""User and authentication related models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """Authenticated user profile from the configured identity provider."""

    id: str = Field(..., description="User's unique identifier")
    display_name: str = Field(..., description="User's display name")
    email: str = Field(..., description="User's email address")
    user_principal_name: str = Field(..., description="User principal name (UPN)")
    tenant_id: str = Field(..., description="Azure AD tenant ID")
    groups: list[str] = Field(default_factory=list, description="User's group memberships")


class TokenInfo(BaseModel):
    """OAuth token information for the application identity provider."""

    access_token: str = Field(..., description="Access token")
    refresh_token: str | None = Field(default=None, description="Refresh token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_at: datetime = Field(..., description="Token expiration time")
    scope: str = Field(..., description="Token scope")
    id_token: str | None = Field(default=None, description="ID token (for user profile fallback)")

    @property
    def is_expired(self) -> bool:
        """Check if the token is expired."""
        return datetime.utcnow() >= self.expires_at


class AuthState(BaseModel):
    """Authentication state for OAuth flow."""

    state: str = Field(..., description="OAuth state parameter")
    code_verifier: str = Field(..., description="PKCE code verifier")
    redirect_uri: str = Field(..., description="Redirect URI")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SessionContext(BaseModel):
    """User session context."""

    session_id: str = Field(..., description="Unique session identifier")
    user: UserProfile = Field(..., description="User profile")
    token_info: TokenInfo | None = Field(
        default=None,
        description="OAuth token information, when supplied by the identity provider",
    )
    selected_org: str | None = Field(default=None, description="Selected organization URL")
    selected_project: str | None = Field(default=None, description="Selected project ID")
    integration_tokens: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Provider tokens, encrypted when stored",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
