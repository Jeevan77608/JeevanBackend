"""
Azure App Service entry point.
This file is used by Azure App Service when no startup command is configured.
It imports the FastAPI app from src.main.

Also ensures the app root is in sys.path so 'src' can be found when Oryx
runs uvicorn from a different working directory (fixes ModuleNotFoundError).
"""

import sys
from pathlib import Path

# Ensure app root is on Python path (required for Oryx Zip Deploy)
_app_root = Path(__file__).resolve().parent
if str(_app_root) not in sys.path:
    sys.path.insert(0, str(_app_root))

# Import the FastAPI app instance
from src.main import app

# Azure App Service will use this as the WSGI application
# For ASGI apps like FastAPI, Azure will use the startup command if configured
# Otherwise, this provides a fallback entry point

__all__ = ["app"]
