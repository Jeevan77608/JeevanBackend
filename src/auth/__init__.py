"""
Authentication module.

Provides Azure AD OAuth 2.0 authentication with PKCE support.
"""

from src.auth.dependencies import (
    get_current_session,
    get_optional_session,
    get_session_manager,
)
from src.auth.oauth import (
    AuthenticationError,
    AuthorizationError,
    AuthorizationState,
    AzureADOAuthProvider,
    PKCEChallenge,
    TokenError,
    TokenValidator,
    UserInfoError,
)

__all__ = [
    # OAuth Provider
    "AzureADOAuthProvider",
    "TokenValidator",
    "PKCEChallenge",
    "AuthorizationState",
    # Exceptions
    "AuthenticationError",
    "AuthorizationError",
    "TokenError",
    "UserInfoError",
    # Dependencies
    "get_session_manager",
    "get_current_session",
    "get_optional_session",
]

