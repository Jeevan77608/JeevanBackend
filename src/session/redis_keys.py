"""Redis key patterns for session-scoped storage."""

from __future__ import annotations

from typing import Final


class RedisKeys:
    """Redis key prefixes and patterns."""

    SESSION: Final[str] = "session:{session_id}"
    DRAFTS: Final[str] = "drafts:{session_id}"
    CONVERSATION: Final[str] = "conversation:{session_id}"
    DRAFT_CONVERSATION: Final[str] = "draft_conversation:{session_id}"
    AUTH_STATE: Final[str] = "auth_state:{state}"
    USER_SESSIONS: Final[str] = "user_sessions:{user_id}"
    FILE_CONTENT: Final[str] = "file:{session_id}:{file_id}"
    BOARD_CONTEXT: Final[str] = "board_context:{session_id}:{project_id}"

    @classmethod
    def session(cls, session_id: str) -> str:
        return cls.SESSION.format(session_id=session_id)

    @classmethod
    def drafts(cls, session_id: str) -> str:
        return cls.DRAFTS.format(session_id=session_id)

    @classmethod
    def conversation(cls, session_id: str) -> str:
        return cls.CONVERSATION.format(session_id=session_id)

    @classmethod
    def draft_conversation(cls, session_id: str) -> str:
        return cls.DRAFT_CONVERSATION.format(session_id=session_id)

    @classmethod
    def auth_state(cls, state: str) -> str:
        return cls.AUTH_STATE.format(state=state)

    @classmethod
    def user_sessions(cls, user_id: str) -> str:
        return cls.USER_SESSIONS.format(user_id=user_id)

    @classmethod
    def file_content(cls, session_id: str, file_id: str) -> str:
        return cls.FILE_CONTENT.format(session_id=session_id, file_id=file_id)

    @classmethod
    def board_context(cls, session_id: str, project_id: str) -> str:
        return cls.BOARD_CONTEXT.format(session_id=session_id, project_id=project_id)
