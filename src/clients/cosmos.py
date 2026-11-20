"""
Azure Cosmos DB client for conversation persistence.

Provides database and container access. Partition key: /user_id.
Only used when CHAT_SAVE_ENABLED=true.

Imports from azure.cosmos are done lazily inside get_cosmos_client() so that
this module can be loaded without azure-cosmos installed; conversation tools
are then registered and only fail at runtime with a clear error if the package
is missing.
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

from src.config.settings import get_settings

if TYPE_CHECKING:
    from azure.cosmos import CosmosClient
    from azure.cosmos.container import ContainerProxy
    from azure.cosmos.database import DatabaseProxy

logger = logging.getLogger(__name__)

# Default TTL on container (seconds); can be overridden per document
DEFAULT_CONTAINER_TTL = -1  # -1 = no default; we set ttl per document

# Cached client/database/container so we don't re-create or re-validate on every request
_cosmos_client: CosmosClient | None = None
_conversations_database: DatabaseProxy | None = None
_conversations_container: ContainerProxy | None = None


def get_cosmos_client() -> CosmosClient | None:
    """Return Cosmos client when chat save is enabled and configured. Cached per process."""
    global _cosmos_client
    if _cosmos_client is not None:
        return _cosmos_client
    try:
        from azure.cosmos import CosmosClient
    except ModuleNotFoundError as e:
        if "azure" in str(e).lower() or "cosmos" in str(e).lower():
            raise RuntimeError(
                "Chat save requires the azure-cosmos package. Install it with: pip install azure-cosmos"
            ) from e
        raise

    settings = get_settings()
    if not settings.chat_save_enabled or not settings.cosmos.endpoint:
        return None
    raw_key = settings.cosmos.key.get_secret_value()
    if not raw_key:
        return None

    def decode_key(k: str) -> str:
        s = k.strip().rstrip("=")
        padded = s + "=" * ((4 - (len(s) % 4)) % 4)
        key_bytes = base64.b64decode(padded, validate=False)
        return base64.b64encode(key_bytes).decode("ascii")

    try:
        key = decode_key(raw_key)
    except Exception as e:
        raise RuntimeError(
            f"Invalid COSMOS_KEY (must be base64): {e}. "
            "Copy the full Primary Key from Azure Portal: Cosmos DB account → Keys."
        ) from e
    _cosmos_client = CosmosClient(settings.cosmos.endpoint, credential=key)
    return _cosmos_client


def get_conversations_database() -> DatabaseProxy | None:
    """Return the conversations database or None if feature disabled. Creates DB if not exists. Cached per process."""
    global _cosmos_client, _conversations_database
    if _conversations_database is not None:
        return _conversations_database
    client = get_cosmos_client()
    if client is None:
        return None
    settings = get_settings()
    _conversations_database = client.create_database_if_not_exists(id=settings.cosmos.database_id)
    return _conversations_database


def get_conversations_container() -> ContainerProxy | None:
    """
    Return the conversations container (partition key: user_id).

    Creates database and container if they do not exist (idempotent). Cached per process for speed.
    """
    from azure.cosmos import PartitionKey

    global _conversations_container
    if _conversations_container is not None:
        return _conversations_container
    database = get_conversations_database()
    if database is None:
        return None
    settings = get_settings()
    create_kw: dict = {
        "id": settings.cosmos.container_id,
        "partition_key": PartitionKey(path="/user_id"),
        "default_ttl": DEFAULT_CONTAINER_TTL,
    }
    if not settings.cosmos.serverless:
        create_kw["offer_throughput"] = 400  # 400 RU minimum for provisioned throughput
    try:
        _conversations_container = database.create_container_if_not_exists(**create_kw)
        return _conversations_container
    except Exception as e:
        logger.exception("create_container_if_not_exists failed: %s", e)
        raise RuntimeError(
            "Could not create Cosmos DB container for conversations. "
            "Ensure your Cosmos DB account exists in Azure and COSMOS_ENDPOINT/COSMOS_KEY are correct. Error: " + str(e)
        ) from e
