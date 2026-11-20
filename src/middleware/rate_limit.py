"""
Rate limiting middleware for Azure DevOps MCP Server.

Provides configurable rate limiting using slowapi with support for:
- Per-user rate limits (authenticated requests)
- Per-IP rate limits (unauthenticated requests)
- Redis-backed storage for distributed rate limiting
- In-memory storage for development

Rate Limit Tiers:
- Standard endpoints: 100 requests/minute
- Tool execution: 30 requests/minute (AI generation is expensive)
- Auth endpoints: 10 requests/minute (prevent brute force)

Example:
    from src.middleware.rate_limit import get_rate_limiter, rate_limit_key_func

    limiter = get_rate_limiter()

    @router.get("/endpoint")
    @limiter.limit("100/minute", key_func=rate_limit_key_func)
    async def endpoint(request: Request):
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


# =============================================================================
# Rate Limit Configuration
# =============================================================================


@dataclass(frozen=True)
class RateLimitConfig:
    """
    Rate limiting configuration for different endpoint types.

    Attributes:
        standard: Default rate limit for most endpoints
        tool_execution: Rate limit for AI tool execution (more restrictive)
        auth: Rate limit for authentication endpoints (most restrictive)
        health: Rate limit for health check endpoints (less restrictive)
    """

    standard: str = "100/minute"
    tool_execution: str = "30/minute"
    auth: str = "10/minute"
    health: str = "200/minute"

    # Burst limits for short-term spikes
    standard_burst: str = "10/second"
    tool_execution_burst: str = "5/second"


# Default configuration
DEFAULT_RATE_LIMITS = RateLimitConfig()


# =============================================================================
# Key Functions
# =============================================================================


def rate_limit_key_func(request: Request) -> str:
    """
    Extract rate limit key from request.

    Uses session ID for authenticated users, falls back to IP address
    for unauthenticated requests. This ensures fair rate limiting
    per user rather than per IP (important for shared IPs).

    Args:
        request: FastAPI request object

    Returns:
        Unique identifier string for rate limiting
    """
    # Try to get session ID from header (authenticated requests)
    session_id = request.headers.get("X-Session-ID")
    if session_id:
        return f"session:{session_id}"

    # Try to get user ID from state (if set by auth middleware)
    if hasattr(request.state, "user_id"):
        return f"user:{request.state.user_id}"

    # Fall back to IP address
    return f"ip:{get_remote_address(request)}"


def rate_limit_key_ip_only(request: Request) -> str:
    """
    Extract IP-based rate limit key.

    Used for auth endpoints where session doesn't exist yet.

    Args:
        request: FastAPI request object

    Returns:
        IP-based identifier string
    """
    return f"ip:{get_remote_address(request)}"


# =============================================================================
# Limiter Factory
# =============================================================================


_limiter_instance: Limiter | None = None


def get_rate_limiter() -> Limiter:
    """
    Get or create the rate limiter instance.

    Creates a singleton Limiter configured based on application settings.
    Uses Redis for distributed rate limiting in production, in-memory
    for development.

    Returns:
        Configured Limiter instance
    """
    global _limiter_instance

    if _limiter_instance is not None:
        return _limiter_instance

    settings = get_settings()

    # Configure storage backend
    if settings.redis_enabled:
        # Use Redis for distributed rate limiting
        storage_uri = settings.redis.url
        logger.info(f"Rate limiter using Redis storage: {storage_uri[:30]}...")
    else:
        # Use in-memory storage for development
        storage_uri = "memory://"
        logger.info("Rate limiter using in-memory storage (development mode)")

    _limiter_instance = Limiter(
        key_func=rate_limit_key_func,
        default_limits=[DEFAULT_RATE_LIMITS.standard],
        storage_uri=storage_uri,
        strategy="fixed-window",  # or "moving-window" for stricter limiting
    )

    return _limiter_instance


def reset_rate_limiter() -> None:
    """Reset the rate limiter instance (useful for testing)."""
    global _limiter_instance
    _limiter_instance = None

