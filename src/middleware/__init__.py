"""
Middleware package for Azure DevOps MCP Server.

This package provides enterprise-grade middleware components:
- Rate limiting: Protect API from abuse
- Request logging: Audit trail for all requests
- Error handling: Standardized error responses
"""

from src.middleware.rate_limit import (
    RateLimitConfig,
    get_rate_limiter,
    rate_limit_key_func,
)

__all__ = [
    "RateLimitConfig",
    "get_rate_limiter",
    "rate_limit_key_func",
]

