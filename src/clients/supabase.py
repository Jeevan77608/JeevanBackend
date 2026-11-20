"""HTTP client for server-side Supabase REST and Storage operations."""

from __future__ import annotations

from typing import Any

import httpx

from src.config.settings import Settings


class SupabaseClient:
    def __init__(self, settings: Settings) -> None:
        self._url = settings.supabase_url.rstrip("/")
        self._key = settings.supabase_service_role_key.get_secret_value()
        if not self._url or not self._key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required for "
                "server-side Supabase operations."
            )

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "apikey": self._key,
            "Authorization": f"Bearer {self._key}",
        }
        if extra:
            headers.update(extra)
        return headers

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        json_body: Any = None,
        headers: dict[str, str] | None = None,
        expected_statuses: tuple[int, ...] = (),
    ) -> httpx.Response:
        return self._request_sync(
            method,
            path,
            params=params,
            json_body=json_body,
            headers=headers,
            expected_statuses=expected_statuses,
        )

    def _request_sync(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None,
        json_body: Any,
        headers: dict[str, str] | None,
        expected_statuses: tuple[int, ...],
    ) -> httpx.Response:
        try:
            with httpx.Client(timeout=15.0) as client:
                response = client.request(
                    method,
                    f"{self._url}{path}",
                    params=params,
                    json=json_body,
                    headers=self._headers(headers),
                )
        except httpx.HTTPError as error:
            raise RuntimeError("Could not reach Supabase.") from error

        if not response.is_success and response.status_code not in expected_statuses:
            raise RuntimeError(
                f"Supabase request failed with status {response.status_code}: "
                f"{response.text[:200]}"
            )
        return response

    async def arequest(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        json_body: Any = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        expected_statuses: tuple[int, ...] = (),
    ) -> httpx.Response:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.request(
                    method,
                    f"{self._url}{path}",
                    params=params,
                    json=json_body,
                    content=content,
                    headers=self._headers(headers),
                )
        except httpx.HTTPError as error:
            raise RuntimeError("Could not reach Supabase.") from error

        if not response.is_success and response.status_code not in expected_statuses:
            raise RuntimeError(
                f"Supabase request failed with status {response.status_code}: "
                f"{response.text[:200]}"
            )
        return response
