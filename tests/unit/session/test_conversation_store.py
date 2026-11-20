"""Tests for src/session/conversation_store.py — conversation history storage."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from src.session.conversation_store import ConversationStore
from src.session.draft_repository import DraftRepository
from src.session.memory_store import InMemoryStore
from src.session.models import ConversationHistory

SESSION_ID = "sess-conv-test-001"


@pytest.fixture()
async def store():
    mem = InMemoryStore()
    await mem.initialize()
    timeout = timedelta(hours=1)
    draft_repo = DraftRepository(mem, timeout)
    conv_store = ConversationStore(mem, timeout, draft_repo)
    yield conv_store, draft_repo, mem
    await mem.close()


class TestConversationStore:
    """Verify ConversationStore operations including draft conversation side-effects."""

    async def test_save_and_get_conversation_round_trip(
        self, store: tuple[ConversationStore, DraftRepository, InMemoryStore]
    ) -> None:
        conv_store, _, _ = store
        conversation = ConversationHistory(session_id=SESSION_ID)
        conversation.add_message("user", "Hello")
        conversation.add_message("assistant", "Hi there!")

        await conv_store.save_conversation(SESSION_ID, conversation)
        result = await conv_store.get_conversation(SESSION_ID)

        assert result is not None
        assert result.session_id == SESSION_ID
        assert len(result.messages) == 2
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "Hello"
        assert result.messages[1].role == "assistant"
        assert result.messages[1].content == "Hi there!"

    async def test_get_conversation_returns_none_when_empty(
        self, store: tuple[ConversationStore, DraftRepository, InMemoryStore]
    ) -> None:
        conv_store, _, _ = store
        result = await conv_store.get_conversation("nonexistent-session")
        assert result is None

    async def test_add_message_creates_new_conversation(
        self, store: tuple[ConversationStore, DraftRepository, InMemoryStore]
    ) -> None:
        conv_store, _, _ = store
        result = await conv_store.add_message(SESSION_ID, "user", "First message")

        assert result is not None
        assert result.session_id == SESSION_ID
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "First message"

    async def test_add_message_appends_to_existing(
        self, store: tuple[ConversationStore, DraftRepository, InMemoryStore]
    ) -> None:
        conv_store, _, _ = store
        await conv_store.add_message(SESSION_ID, "user", "Message 1")
        result = await conv_store.add_message(SESSION_ID, "assistant", "Message 2")

        assert result is not None
        assert len(result.messages) == 2
        assert result.messages[0].content == "Message 1"
        assert result.messages[1].content == "Message 2"

    async def test_clear_conversation_keeps_system_messages(
        self, store: tuple[ConversationStore, DraftRepository, InMemoryStore]
    ) -> None:
        conv_store, _, _ = store
        conversation = ConversationHistory(session_id=SESSION_ID)
        conversation.add_message("system", "You are a helpful assistant.")
        conversation.add_message("user", "Hello")
        conversation.add_message("assistant", "Hi!")
        await conv_store.save_conversation(SESSION_ID, conversation)

        await conv_store.clear_conversation(SESSION_ID)
        result = await conv_store.get_conversation(SESSION_ID)

        assert result is not None
        assert len(result.messages) == 1
        assert result.messages[0].role == "system"
        assert result.messages[0].content == "You are a helpful assistant."

    async def test_save_and_get_draft_conversation(
        self, store: tuple[ConversationStore, DraftRepository, InMemoryStore]
    ) -> None:
        conv_store, _, _ = store
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Draft msg 1"},
            {"role": "assistant", "content": "Draft reply"},
        ]
        await conv_store.save_draft_conversation(SESSION_ID, messages, title="Draft Chat")

        result = await conv_store.get_draft_conversation(SESSION_ID)
        assert result is not None
        assert result["title"] == "Draft Chat"
        assert len(result["messages_snapshot"]) == 2
        assert result["messages_snapshot"][0]["content"] == "Draft msg 1"

    async def test_get_draft_conversation_returns_none(
        self, store: tuple[ConversationStore, DraftRepository, InMemoryStore]
    ) -> None:
        conv_store, _, _ = store
        result = await conv_store.get_draft_conversation("nonexistent-session")
        assert result is None

    async def test_clear_draft_conversation_also_deletes_drafts(
        self, store: tuple[ConversationStore, DraftRepository, InMemoryStore]
    ) -> None:
        conv_store, draft_repo, _ = store

        # Save a draft conversation
        messages: list[dict[str, Any]] = [{"role": "user", "content": "Temp"}]
        await conv_store.save_draft_conversation(SESSION_ID, messages, title="Temp")

        # Save some drafts via the draft repository
        from src.models.work_items import DraftCollection, DraftWorkItem

        draft = DraftWorkItem(
            draft_id="draft-001",
            work_item_type="User Story",
            title="To be deleted",
        )
        collection = DraftCollection(
            session_id=SESSION_ID,
            project_id="TestProject",
            items=[draft],
        )
        await draft_repo.save_drafts(SESSION_ID, collection)

        # Clear draft conversation — should also remove drafts
        await conv_store.clear_draft_conversation(SESSION_ID)

        assert await conv_store.get_draft_conversation(SESSION_ID) is None
        assert await draft_repo.get_drafts(SESSION_ID) is None
