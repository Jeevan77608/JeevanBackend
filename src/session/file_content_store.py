"""File content storage for uploaded files used as context."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from src.clients.supabase import SupabaseClient
from src.config.settings import Settings
from src.session.base_store import BaseStore, StoreType
from src.session.redis_keys import RedisKeys

logger = logging.getLogger(__name__)


class FileContentStore(BaseStore):
    """Repository for uploaded file content storage."""

    def __init__(
        self,
        store: StoreType,
        idle_timeout: timedelta,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(store, idle_timeout)
        self._supabase = (
            SupabaseClient(settings)
            if settings and settings.supabase_storage_enabled
            else None
        )
        self._bucket = settings.supabase_storage_bucket if settings else ""

    def _object_path(self, session_id: str, file_id: str) -> str:
        safe_file_id = hashlib.sha256(file_id.encode("utf-8")).hexdigest()
        return f"{quote(session_id, safe='')}/{safe_file_id}.txt"

    async def store_file_content(
        self,
        session_id: str,
        file_id: str,
        content: str,
    ) -> None:
        """Store file content for use in work item generation."""
        if self._supabase:
            path = self._object_path(session_id, file_id)
            encoded_path = quote(f"{self._bucket}/{path}", safe="/")
            await self._supabase.arequest(
                "POST",
                f"/storage/v1/object/{encoded_path}",
                content=content.encode("utf-8"),
                headers={
                    "Content-Type": "text/plain; charset=utf-8",
                    "x-upsert": "true",
                },
            )
            expires_at = datetime.now(timezone.utc) + self._idle_timeout
            try:
                await self._supabase.arequest(
                    "POST",
                    "/rest/v1/session_files",
                    params={"on_conflict": "session_id,file_id"},
                    json_body=[
                        {
                            "session_id": session_id,
                            "file_id": file_id,
                            "storage_path": path,
                            "expires_at": expires_at.isoformat(),
                        }
                    ],
                    headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
                )
            except RuntimeError:
                await self._supabase.arequest(
                    "DELETE",
                    f"/storage/v1/object/{quote(self._bucket, safe='')}",
                    json_body={"prefixes": [path]},
                )
                raise
            return

        r = self._store
        key = RedisKeys.file_content(session_id, file_id)

        await r.setex(
            key,
            self._idle_timeout,
            content,
        )
        logger.debug(
            "Stored file content for session %s, file %s", session_id, file_id
        )

    async def get_file_content(
        self,
        session_id: str,
        file_id: str,
    ) -> str | None:
        """Retrieve stored file content."""
        if self._supabase:
            path = self._object_path(session_id, file_id)
            metadata = await self._supabase.arequest(
                "GET",
                "/rest/v1/session_files",
                params={
                    "select": "storage_path",
                    "session_id": f"eq.{session_id}",
                    "storage_path": f"eq.{path}",
                    "expires_at": f"gt.{datetime.now(timezone.utc).isoformat()}",
                    "limit": "1",
                },
            )
            rows = metadata.json()
            if not rows:
                return None

            response = await self._supabase.arequest(
                "GET",
                f"/storage/v1/object/{quote(f'{self._bucket}/{path}', safe='/')}",
                expected_statuses=(404,),
            )
            if response.status_code == 404:
                return None
            return response.content.decode("utf-8")

        r = self._store
        key = RedisKeys.file_content(session_id, file_id)

        content = await r.get(key)
        if content:
            return content.decode("utf-8") if isinstance(content, bytes) else content
        return None

    async def list_file_contents(self, session_id: str) -> list[str]:
        """List all file IDs stored for a session."""
        if self._supabase:
            response = await self._supabase.arequest(
                "GET",
                "/rest/v1/session_files",
                params={
                    "select": "file_id",
                    "session_id": f"eq.{session_id}",
                    "expires_at": f"gt.{datetime.now(timezone.utc).isoformat()}",
                },
            )
            return [item["file_id"] for item in response.json()]

        r = self._store
        pattern = f"file:{session_id}:*"

        keys = await r.keys(pattern)
        return [
            (key.decode("utf-8") if isinstance(key, bytes) else key).split(":")[-1]
            for key in keys
        ]

    async def delete_session_files(self, session_id: str) -> None:
        """Delete all private Storage objects associated with a session."""
        if not self._supabase:
            r = self._store
            keys = await r.keys(f"file:{session_id}:*")
            if keys:
                await r.delete(*keys)
            return

        response = await self._supabase.arequest(
            "GET",
            "/rest/v1/session_files",
            params={
                "select": "storage_path",
                "session_id": f"eq.{session_id}",
            },
        )
        paths = [item["storage_path"] for item in response.json()]
        if paths:
            await self._supabase.arequest(
                "DELETE",
                f"/storage/v1/object/{quote(self._bucket, safe='')}",
                json_body={"prefixes": paths},
            )
        await self._supabase.arequest(
            "DELETE",
            "/rest/v1/session_files",
            params={"session_id": f"eq.{session_id}"},
        )

    async def cleanup_expired_files(self) -> None:
        """Remove expired file objects and their metadata rows."""
        if not self._supabase:
            return

        response = await self._supabase.arequest(
            "GET",
            "/rest/v1/session_files",
            params={
                "select": "session_id,file_id,storage_path",
                "expires_at": f"lte.{datetime.now(timezone.utc).isoformat()}",
            },
        )
        expired = response.json()
        if not expired:
            return

        await self._supabase.arequest(
            "DELETE",
            f"/storage/v1/object/{quote(self._bucket, safe='')}",
            json_body={"prefixes": [item["storage_path"] for item in expired]},
        )
        for item in expired:
            await self._supabase.arequest(
                "DELETE",
                "/rest/v1/session_files",
                params={
                    "session_id": f"eq.{item['session_id']}",
                    "file_id": f"eq.{item['file_id']}",
                },
            )
