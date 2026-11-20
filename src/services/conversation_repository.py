"""
Conversation repository: Cosmos DB CRUD for chat history.

All operations are scoped by user_id from context (session/JWT); never trust client-supplied user_id.
TTL is set from CHAT_RETENTION_DAYS for automatic expiry.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from src.clients.cosmos import get_conversations_container
from src.config.settings import get_settings
from src.models.conversation import ConversationDocument

logger = logging.getLogger(__name__)


def _ttl_seconds() -> int:
    """TTL in seconds from config (retention days)."""
    return get_settings().chat_retention_days * 86400


def _uses_supabase() -> bool:
    return get_settings().auth_provider == "supabase"


def _supabase_request(
    method: str,
    *,
    params: dict[str, str] | None = None,
    json_body: Any = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    settings = get_settings()
    url = settings.supabase_url.rstrip("/")
    key = settings.supabase_service_role_key.get_secret_value()
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required for "
            "Supabase conversation storage."
        )

    request_headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }
    if headers:
        request_headers.update(headers)

    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.request(
                method,
                f"{url}/rest/v1/conversations",
                params=params,
                json=json_body,
                headers=request_headers,
            )
    except httpx.HTTPError as error:
        raise RuntimeError("Could not reach Supabase conversation storage.") from error

    if not response.is_success:
        raise RuntimeError(
            f"Supabase conversation request failed with status "
            f"{response.status_code}: {response.text[:200]}"
        )
    return response


def _purge_expired_supabase_conversations() -> None:
    _supabase_request(
        "DELETE",
        params={"expires_at": f"lte.{datetime.now(timezone.utc).isoformat()}"},
    )


def _save_supabase_conversation(
    user_id: str,
    *,
    conversation_id: str | None,
    session_id: str | None,
    title: str,
    markdown: str,
    messages_snapshot: list[dict[str, Any]] | None,
    context_snapshot: dict[str, Any] | None,
    drafts_snapshot: list[dict[str, Any]] | None,
    existing_doc: ConversationDocument | None,
) -> ConversationDocument:
    now = datetime.now(timezone.utc)
    ttl = _ttl_seconds()
    existing = None
    if conversation_id:
        existing = (
            existing_doc
            if existing_doc
            and existing_doc.id == conversation_id
            and existing_doc.user_id == user_id
            else get_conversation(user_id, conversation_id)
        )

    if existing:
        doc = ConversationDocument(
            id=existing.id,
            user_id=user_id,
            session_id=session_id or existing.session_id,
            title=title or existing.title,
            markdown=markdown or existing.markdown,
            messages_snapshot=(
                messages_snapshot
                if messages_snapshot is not None
                else existing.messages_snapshot
            ),
            context_snapshot=(
                context_snapshot
                if context_snapshot is not None
                else existing.context_snapshot
            ),
            drafts_snapshot=(
                drafts_snapshot
                if drafts_snapshot is not None
                else existing.drafts_snapshot
            ),
            created_at=existing.created_at,
            updated_at=now,
            ttl=ttl,
        )
    else:
        doc = ConversationDocument(
            id=conversation_id or str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            title=title,
            markdown=markdown,
            messages_snapshot=messages_snapshot or [],
            context_snapshot=context_snapshot,
            drafts_snapshot=drafts_snapshot,
            created_at=now,
            updated_at=now,
            ttl=ttl,
        )

    payload = doc.model_dump(mode="json")
    payload.pop("ttl", None)
    payload["expires_at"] = (
        now + timedelta(seconds=ttl)
    ).isoformat()
    response = _supabase_request(
        "POST",
        params={"on_conflict": "user_id,id"},
        json_body=[payload],
        headers={"Prefer": "resolution=merge-duplicates,return=representation"},
    )
    rows = response.json()
    return ConversationDocument.model_validate(rows[0]) if rows else doc


def save_conversation(
    user_id: str,
    *,
    conversation_id: str | None = None,
    session_id: str | None = None,
    title: str = "",
    markdown: str = "",
    messages_snapshot: list[dict[str, Any]] | None = None,
    context_snapshot: dict[str, Any] | None = None,
    drafts_snapshot: list[dict[str, Any]] | None = None,
    existing_doc: ConversationDocument | None = None,
) -> ConversationDocument:
    """
    Create or update a conversation. Always scoped by user_id (from auth).

    If conversation_id is provided and exists for this user, update; otherwise create new.
    Pass existing_doc when already loaded to avoid a duplicate read (e.g. from update_conversation_drafts).
    Returns the saved document (with id, created_at, updated_at).
    """
    if _uses_supabase():
        return _save_supabase_conversation(
            user_id,
            conversation_id=conversation_id,
            session_id=session_id,
            title=title,
            markdown=markdown,
            messages_snapshot=messages_snapshot,
            context_snapshot=context_snapshot,
            drafts_snapshot=drafts_snapshot,
            existing_doc=existing_doc,
        )

    container = get_conversations_container()
    if container is None:
        raise RuntimeError("Conversation storage is not configured (CHAT_SAVE_ENABLED/Cosmos)")

    now = datetime.now(timezone.utc)
    ttl = _ttl_seconds()

    if conversation_id:
        existing = existing_doc if existing_doc is not None and existing_doc.id == conversation_id else get_conversation(user_id, conversation_id)
        if existing:
            # Support existing documents that may lack context_snapshot/drafts_snapshot (old schema)
            existing_ctx = getattr(existing, "context_snapshot", None)
            existing_drafts = getattr(existing, "drafts_snapshot", None)
            doc = ConversationDocument(
                id=existing.id,
                user_id=user_id,
                session_id=session_id or existing.session_id,
                title=title or existing.title,
                markdown=markdown or existing.markdown,
                messages_snapshot=messages_snapshot if messages_snapshot is not None else existing.messages_snapshot,
                context_snapshot=context_snapshot if context_snapshot is not None else existing_ctx,
                drafts_snapshot=drafts_snapshot if drafts_snapshot is not None else existing_drafts,
                created_at=existing.created_at,
                updated_at=now,
                ttl=ttl,
            )
        else:
            doc = ConversationDocument(
                id=conversation_id,
                user_id=user_id,
                session_id=session_id,
                title=title,
                markdown=markdown,
                messages_snapshot=messages_snapshot or [],
                context_snapshot=context_snapshot,
                drafts_snapshot=drafts_snapshot,
                created_at=now,
                updated_at=now,
                ttl=ttl,
            )
    else:
        import uuid
        new_id = str(uuid.uuid4())
        doc = ConversationDocument(
            id=new_id,
            user_id=user_id,
            session_id=session_id,
            title=title,
            markdown=markdown,
            messages_snapshot=messages_snapshot or [],
            context_snapshot=context_snapshot,
            drafts_snapshot=drafts_snapshot,
            created_at=now,
            updated_at=now,
            ttl=ttl,
        )

    item = doc.to_cosmos_item()
    container.upsert_item(body=item)
    return doc


def delete_conversation(user_id: str, conversation_id: str) -> bool:
    """
    Delete a conversation by id. Returns True if deleted, False if not found or not owned by user_id.
    """
    if _uses_supabase():
        response = _supabase_request(
            "DELETE",
            params={
                "id": f"eq.{conversation_id}",
                "user_id": f"eq.{user_id}",
            },
            headers={"Prefer": "return=representation"},
        )
        return bool(response.json())

    container = get_conversations_container()
    if container is None:
        return False
    existing = get_conversation(user_id, conversation_id)
    if existing is None:
        return False
    try:
        container.delete_item(item=conversation_id, partition_key=user_id)
        return True
    except Exception as e:
        logger.debug("delete_conversation %s: %s", conversation_id, e)
        return False


def update_conversation_drafts(
    user_id: str,
    conversation_id: str,
    drafts_snapshot: list[dict[str, Any]],
) -> bool:
    """
    Update only the drafts_snapshot of an existing conversation.
    Returns True if updated, False if conversation not found or not owned by user.
    Use this to persist draft changes (create/update/delete) immediately so they survive reload and chat switch.
    """
    existing = get_conversation(user_id, conversation_id)
    if existing is None:
        return False
    save_conversation(
        user_id,
        conversation_id=conversation_id,
        session_id=existing.session_id,
        title=existing.title,
        markdown=existing.markdown,
        messages_snapshot=existing.messages_snapshot,
        context_snapshot=getattr(existing, "context_snapshot", None),
        drafts_snapshot=drafts_snapshot,
        existing_doc=existing,
    )
    return True


def get_conversation(user_id: str, conversation_id: str) -> ConversationDocument | None:
    """
    Get a conversation by id. Returns None if not found or not owned by user_id.
    """
    if _uses_supabase():
        response = _supabase_request(
            "GET",
            params={
                "select": "*",
                "id": f"eq.{conversation_id}",
                "user_id": f"eq.{user_id}",
                "expires_at": f"gt.{datetime.now(timezone.utc).isoformat()}",
                "limit": "1",
            },
        )
        rows = response.json()
        return ConversationDocument.model_validate(rows[0]) if rows else None

    container = get_conversations_container()
    if container is None:
        return None
    try:
        item = container.read_item(item=conversation_id, partition_key=user_id)
        return ConversationDocument.from_cosmos_item(item)
    except Exception as e:
        logger.debug("get_conversation %s: %s", conversation_id, e)
        return None


def list_conversations(user_id: str, limit: int = 50, days: int | None = None) -> list[ConversationDocument]:
    """
    List conversations for user_id, newest first.

    Uses a projection query (id, title, updated_at, created_at only) so we don't pull
    markdown or messages_snapshot—much faster and fewer RUs.

    Args:
        user_id: Partition key (from auth).
        limit: Maximum number of results.
        days: When provided, only return conversations updated within this many days.
    """
    if _uses_supabase():
        now = datetime.now(timezone.utc)
        _purge_expired_supabase_conversations()
        params = {
            "select": "id,user_id,title,created_at,updated_at",
            "user_id": f"eq.{user_id}",
            "expires_at": f"gt.{now.isoformat()}",
            "order": "updated_at.desc",
            "limit": str(max(1, min(limit, 100))),
        }
        if days is not None:
            params["updated_at"] = f"gte.{(now - timedelta(days=days)).isoformat()}"
        response = _supabase_request("GET", params=params)
        return [
            ConversationDocument.model_validate(row)
            for row in response.json()
        ]

    container = get_conversations_container()
    if container is None:
        return []

    parameters: list[dict[str, Any]] = [
        {"name": "@user_id", "value": user_id},
        {"name": "@limit", "value": limit},
    ]

    if days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        parameters.append({"name": "@cutoff", "value": cutoff.isoformat()})
        query = (
            "SELECT c.id, c.title, c.updated_at, c.created_at FROM c "
            "WHERE c.user_id = @user_id AND c.updated_at >= @cutoff "
            "ORDER BY c.updated_at DESC OFFSET 0 LIMIT @limit"
        )
    else:
        query = (
            "SELECT c.id, c.title, c.updated_at, c.created_at FROM c "
            "WHERE c.user_id = @user_id ORDER BY c.updated_at DESC OFFSET 0 LIMIT @limit"
        )

    items = list(
        container.query_items(
            query=query,
            parameters=parameters,
            partition_key=user_id,
        )
    )
    now = datetime.now(timezone.utc)
    result: list[ConversationDocument] = []
    for i in items:
        created = i.get("created_at")
        updated = i.get("updated_at")
        if isinstance(created, str):
            created = datetime.fromisoformat(created.replace("Z", "+00:00"))
        if isinstance(updated, str):
            updated = datetime.fromisoformat(updated.replace("Z", "+00:00"))
        if created is None:
            created = updated or now
        if updated is None:
            updated = now
        result.append(
            ConversationDocument(
                id=i["id"],
                user_id=user_id,
                title=i.get("title") or "",
                markdown="",
                messages_snapshot=[],
                created_at=created,
                updated_at=updated,
            )
        )
    return result
