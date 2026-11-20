"""
Main FastAPI application entry point.

Azure DevOps MCP Server - AI-powered work item generation and management.

This module serves as the entry point for the FastAPI application, providing:
- Application lifecycle management (startup/shutdown)
- Middleware configuration (CORS, rate limiting, error handling)
- Route registration for REST API and WebSocket endpoints
- OpenAPI/Swagger documentation configuration

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    FastAPI Application                       │
    ├─────────────────────────────────────────────────────────────┤
    │  Middleware Stack:                                           │
    │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
    │  │    CORS     │→│ Rate Limit  │→│    Error Handler        ││
    │  └─────────────┘ └─────────────┘ └─────────────────────────┘│
    ├─────────────────────────────────────────────────────────────┤
    │  Routes:                                                     │
    │  /api/v1/...  - REST API endpoints                          │
    │  /ws/mcp      - WebSocket MCP endpoint                      │
    │  /docs        - Swagger UI (development only)               │
    └─────────────────────────────────────────────────────────────┘

Usage:
    # Development
    uvicorn src.main:app --reload --port 8000

    # Production (use startup.sh so worker count matches REDIS_ENABLED)
    # - REDIS_ENABLED=false → 1 worker (in-memory store is per-process)
    # - REDIS_ENABLED unset or true → 4 workers (matches Python default redis_enabled=True)
    uvicorn src.main:app --host 0.0.0.0 --port $PORT --workers 4  # or 1 if Redis disabled

Environment Variables:
    APP_ENV: development|staging|production
    AUTH_ENABLED: true|false (OAuth vs PAT authentication)
    REDIS_ENABLED: true|false (Redis vs in-memory session storage)
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.api.routes import router as api_router
from src.api.websocket import MCPWebSocketHandler, connection_manager
from src.auth.oauth import AzureADOAuthProvider
from src.config.settings import get_settings
from src.middleware.rate_limit import get_rate_limiter
from src.session.manager import SessionManager

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Application Lifespan
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager for startup and shutdown events.

    Handles initialization and cleanup of:
    - Session manager (Redis or in-memory)
    - OAuth provider (when auth enabled)
    - WebSocket connection manager
    - Rate limiter

    Args:
        app: FastAPI application instance

    Yields:
        None - control returns to FastAPI for request handling
    """
    settings = get_settings()

    # Startup banner
    logger.info("=" * 60)
    logger.info("Jeevan AI Chatbot backend starting")
    logger.info("=" * 60)
    logger.info(f"Environment: {settings.app_env}")
    logger.info(f"Auth Enabled: {settings.auth_enabled}")
    logger.info(f"Redis Enabled: {settings.redis_enabled}")
    logger.info("=" * 60)

    # Initialize session manager (uses Redis or in-memory based on settings)
    session_manager = SessionManager(settings)
    await session_manager.initialize()
    app.state.session_manager = session_manager
    logger.info("Session manager initialized")

    # Initialize OAuth provider only if auth is enabled
    if settings.auth_enabled:
        if settings.auth_provider == "google":
            from src.auth.google_oauth import GoogleOAuthProvider

            app.state.google_oauth_provider = GoogleOAuthProvider(settings)
            app.state.oauth_provider = None
            logger.info("Google OAuth provider initialized")
        elif settings.auth_provider == "azure_ad":
            if settings.azure_ad is None:
                raise RuntimeError("Azure AD settings required when AUTH_ENABLED=true")
            oauth_provider = AzureADOAuthProvider(settings)
            app.state.oauth_provider = oauth_provider
            app.state.google_oauth_provider = None
            logger.info("Azure AD OAuth provider initialized")
        else:
            app.state.oauth_provider = None
            app.state.google_oauth_provider = None
            logger.info("Supabase Auth will provide user sign-in")
    else:
        app.state.oauth_provider = None
        app.state.google_oauth_provider = None
        logger.warning(
            "⚠️  Authentication DISABLED - using mock session. "
            "This should only be used for local development!"
        )

    # Initialize WebSocket handler
    ws_handler = MCPWebSocketHandler(session_manager, connection_manager)
    app.state.ws_handler = ws_handler
    logger.info("WebSocket handler initialized")

    logger.info("✓ Application startup complete")
    logger.info("=" * 60)

    yield

    # Cleanup on shutdown
    logger.info("=" * 60)
    logger.info("Shutting down Jeevan AI Chatbot backend")
    await session_manager.close()
    logger.info("✓ Application shutdown complete")
    logger.info("=" * 60)


# =============================================================================
# Application Factory
# =============================================================================


# OpenAPI tags for Swagger documentation
OPENAPI_TAGS = [
    {
        "name": "health",
        "description": "Health check and monitoring endpoints",
    },
    {
        "name": "auth",
        "description": "OAuth 2.0 authentication with the configured identity provider",
    },
    {
        "name": "session",
        "description": "Session management and user context",
    },
    {
        "name": "tools",
        "description": "MCP tool discovery and execution",
    },
    {
        "name": "drafts",
        "description": "Draft work item management",
    },
    {
        "name": "jira",
        "description": "Per-user Jira Cloud connection and issue management",
    },
]


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    This factory function creates a fully configured FastAPI application with:
    - OpenAPI/Swagger documentation
    - CORS middleware
    - Rate limiting middleware
    - Error handlers
    - API routes
    - WebSocket endpoints

    Returns:
        Configured FastAPI application instance
    """
    settings = get_settings()

    # Create FastAPI app with enhanced OpenAPI documentation
    app = FastAPI(
        title=settings.app_name,
        description="""
## AI-Powered Jira and Azure DevOps Work Management

This MCP (Model Context Protocol) server provides AI-powered tools for managing
Jira Cloud issues and Azure DevOps work items. It enables AI assistants to:

- 🔍 **Query** Jira issues and supported Azure DevOps work items
- ✨ **Generate** and refine work-item drafts using AI
- 🚀 **Create** reviewed drafts in the connected project

### Authentication

User sign-in is handled by Supabase Auth. Jira access is authorized separately
for each user through Atlassian OAuth 2.0.

### Rate Limits

- Standard endpoints: 100 requests/minute
- Tool execution: 30 requests/minute
- Auth endpoints: 10 requests/minute
        """,
        version="1.0.0",
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        openapi_url="/openapi.json" if settings.is_development else None,
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
        contact={
            "name": "API Support",
            "email": "support@example.com",
        },
    )

    # -------------------------------------------------------------------------
    # Middleware Configuration
    # -------------------------------------------------------------------------

    # Rate Limiting
    limiter = get_rate_limiter()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS (Cross-Origin Resource Sharing)
    # CORS_ORIGINS must be set via environment variable (comma-separated list)
    # Empty list means no CORS allowed (secure default)
    cors_origins = settings.get_cors_origins_list()

    # Log CORS configuration for debugging
    logger.info(f"🌐 CORS Configuration: APP_ENV={settings.app_env}, CORS_ORIGINS={settings.cors_origins}")
    logger.info(f"🌐 Parsed CORS origins: {cors_origins}")

    # In development, allow localhost:3000 by default if CORS_ORIGINS is not set
    if settings.is_development and not cors_origins:
        cors_origins = ["http://localhost:3000"]
        logger.info("🔧 Development mode: Allowing CORS from http://localhost:3000")

    if settings.is_production and not cors_origins:
        logger.error(
            "❌ CORS_ORIGINS not set in production! CORS requests will be blocked. "
            "Set CORS_ORIGINS environment variable with comma-separated frontend URLs."
        )
    else:
        logger.info(f"✅ CORS configured to allow origins: {cors_origins}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Session-ID", "X-Internal-Key", "Cookie"],
        expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
    )

    # -------------------------------------------------------------------------
    # Route Registration
    # -------------------------------------------------------------------------

    # REST API routes
    app.include_router(api_router, prefix="/api/v1")

    # WebSocket endpoint for MCP protocol
    @app.websocket("/ws/mcp")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """
        WebSocket endpoint for MCP (Model Context Protocol) communication.

        Provides real-time bidirectional communication for AI assistants
        to interact with Azure DevOps tools.
        """
        connection_id = str(uuid.uuid4())
        ws_handler: MCPWebSocketHandler = app.state.ws_handler
        await ws_handler.handle_connection(websocket, connection_id)

    # Root endpoint with API info
    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        """Root endpoint returning API information."""
        return {
            "name": settings.app_name,
            "version": "1.0.0",
            "docs": "/docs" if settings.is_development else None,
            "health": "/api/v1/health",
        }

    return app


# =============================================================================
# Application Instance
# =============================================================================

app = create_app()


# =============================================================================
# Development Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
