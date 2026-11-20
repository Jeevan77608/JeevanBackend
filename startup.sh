#!/bin/bash
# Azure App Service startup script for FastAPI backend
# Use uvicorn directly - simpler and works better with FastAPI

# Azure App Service will set PORT environment variable
PORT=${PORT:-8000}

# Use application:app (not src.main:app) - application.py adds app root to sys.path
# so 'src' module is found when Oryx runs from a different working directory.
# When REDIS_ENABLED=false, session storage is in-memory per process.
# Multiple workers would each have isolated InMemoryStore -> session loss across workers.
if [ "${REDIS_ENABLED}" = "false" ]; then
  exec uvicorn application:app --host 0.0.0.0 --port $PORT --workers 1
else
  exec uvicorn application:app --host 0.0.0.0 --port $PORT --workers 4
fi
