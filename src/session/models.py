"""
Session-related models following OWASP Session Management guidelines.

This module defines models for:
- Session fingerprinting (binding sessions to clients)
- Session state and lifecycle
- Conversation history for AI interactions
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SessionState(str, Enum):
    """Session lifecycle states."""

    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    LOCKED = "locked"  # Too many failed attempts


class SessionFingerprint(BaseModel):
    """
    Session fingerprint for client binding (OWASP recommendation).

    Helps detect session hijacking by binding sessions to client characteristics.
    We hash sensitive values to avoid storing raw data.
    """

    user_agent_hash: str = Field(..., description="SHA-256 hash of User-Agent")
    ip_hash: str | None = Field(
        default=None,
        description="SHA-256 hash of client IP (optional, can change with mobile)",
    )
    accept_language: str | None = Field(
        default=None,
        description="Accept-Language header value",
    )

    @classmethod
    def create(
        cls,
        user_agent: str,
        client_ip: str | None = None,
        accept_language: str | None = None,
    ) -> SessionFingerprint:
        """Create fingerprint from raw values (hashes sensitive data)."""
        ua_hash = hashlib.sha256(user_agent.encode()).hexdigest()
        ip_hash = (
            hashlib.sha256(client_ip.encode()).hexdigest() if client_ip else None
        )
        return cls(
            user_agent_hash=ua_hash,
            ip_hash=ip_hash,
            accept_language=accept_language,
        )

    def matches(
        self,
        other: SessionFingerprint,
        *,
        strict_ip: bool = False,
    ) -> bool:
        """
        Check if fingerprints match.

        Args:
            other: Fingerprint to compare
            strict_ip: If True, IP hash must also match

        Returns:
            True if fingerprints match
        """
        if self.user_agent_hash != other.user_agent_hash:
            return False
        if strict_ip and self.ip_hash != other.ip_hash:
            return False
        return True


class ConversationMessage(BaseModel):
    """A single message in the conversation history."""

    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Message timestamp",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (tool calls, etc.)",
    )

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate message role."""
        allowed_roles = {"user", "assistant", "system", "tool"}
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of: {allowed_roles}")
        return v


class ConversationHistory(BaseModel):
    """Conversation history for AI context."""

    session_id: str = Field(..., description="Session ID")
    messages: list[ConversationMessage] = Field(
        default_factory=list,
        description="Ordered list of messages",
    )
    max_messages: int = Field(
        default=50,
        description="Maximum messages to retain",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def add_message(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a message and enforce max limit."""
        message = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)

        # Trim old messages if exceeding limit (keep system messages)
        if len(self.messages) > self.max_messages:
            system_msgs = [m for m in self.messages if m.role == "system"]
            other_msgs = [m for m in self.messages if m.role != "system"]
            keep_count = self.max_messages - len(system_msgs)
            self.messages = system_msgs + other_msgs[-keep_count:]

    def get_context_window(self, max_tokens: int = 4000) -> list[dict[str, str]]:
        """Get messages formatted for AI context, respecting token limit."""
        # Simple estimation: ~4 chars per token
        result = []
        total_chars = 0
        max_chars = max_tokens * 4

        for msg in reversed(self.messages):
            msg_chars = len(msg.content)
            if total_chars + msg_chars > max_chars:
                break
            result.insert(0, {"role": msg.role, "content": msg.content})
            total_chars += msg_chars

        return result

    def clear(self) -> None:
        """Clear conversation history except system messages."""
        self.messages = [m for m in self.messages if m.role == "system"]
        self.updated_at = datetime.now(timezone.utc)

