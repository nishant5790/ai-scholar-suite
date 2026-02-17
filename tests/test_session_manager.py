"""Unit tests for the SessionManager."""

import uuid
from datetime import datetime, timezone

import pytest

from src.core.session_manager import PaperSession, SessionManager
from src.models.schemas import PaperState


@pytest.fixture
def manager():
    return SessionManager()


class TestPaperSession:
    def test_default_fields(self):
        session = PaperSession(session_id="test-id")
        assert session.session_id == "test-id"
        assert session.agent is None
        assert isinstance(session.paper_state, PaperState)
        assert isinstance(session.created_at, datetime)

    def test_custom_agent_field(self):
        session = PaperSession(session_id="test-id", agent="mock-agent")
        assert session.agent == "mock-agent"

    def test_created_at_is_utc(self):
        session = PaperSession(session_id="test-id")
        assert session.created_at.tzinfo is not None


class TestCreateSession:
    def test_returns_string_id(self, manager):
        session_id = manager.create_session()
        assert isinstance(session_id, str)

    def test_returns_valid_uuid(self, manager):
        session_id = manager.create_session()
        parsed = uuid.UUID(session_id)
        assert str(parsed) == session_id

    def test_creates_retrievable_session(self, manager):
        session_id = manager.create_session()
        session = manager.get_session(session_id)
        assert session.session_id == session_id

    def test_unique_ids(self, manager):
        ids = {manager.create_session() for _ in range(10)}
        assert len(ids) == 10

    def test_new_session_has_empty_paper_state(self, manager):
        session_id = manager.create_session()
        session = manager.get_session(session_id)
        assert session.paper_state.title == ""
        assert session.paper_state.topic == ""
        assert session.agent is None


class TestGetSession:
    def test_returns_correct_session(self, manager):
        id1 = manager.create_session()
        id2 = manager.create_session()
        assert manager.get_session(id1).session_id == id1
        assert manager.get_session(id2).session_id == id2

    def test_nonexistent_session_raises_key_error(self, manager):
        with pytest.raises(KeyError, match="Session not found"):
            manager.get_session("nonexistent-id")

    def test_returns_same_object(self, manager):
        session_id = manager.create_session()
        s1 = manager.get_session(session_id)
        s2 = manager.get_session(session_id)
        assert s1 is s2


class TestDeleteSession:
    def test_delete_existing_session(self, manager):
        session_id = manager.create_session()
        manager.delete_session(session_id)
        with pytest.raises(KeyError):
            manager.get_session(session_id)

    def test_delete_nonexistent_session_raises_key_error(self, manager):
        with pytest.raises(KeyError, match="Session not found"):
            manager.delete_session("nonexistent-id")

    def test_delete_does_not_affect_other_sessions(self, manager):
        id1 = manager.create_session()
        id2 = manager.create_session()
        manager.delete_session(id1)
        session = manager.get_session(id2)
        assert session.session_id == id2
