"""
External service clients.

Provides clients for:
- Azure DevOps REST API v7.0
- Azure OpenAI GPT-4
"""

from src.clients.devops import (
    ApiResponse,
    AuthenticationError,
    AzureDevOpsClient,
    AzureDevOpsError,
    NotFoundError,
    RateLimitError,
    TokenBucket,
    ValidationError,
)

__all__ = [
    # Azure DevOps
    "AzureDevOpsClient",
    "AzureDevOpsError",
    "AuthenticationError",
    "NotFoundError",
    "RateLimitError",
    "ValidationError",
    "ApiResponse",
    "TokenBucket",
]

