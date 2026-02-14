"""Session manager for managing per-session agent instances and paper state."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from src.models.schemas import PaperState


@dataclass
class PaperSession:
    """Represents a single paper-writing session."""

    session_id: str
    agent: Optional[Any] = None  # Will be AgentExecutor once agent is implemented
    paper_state: PaperState = field(default_factory=PaperState)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SessionManager:
    """Manages per-session agent instances and paper state using in-memory storage."""

    def __init__(self) -> None:
        self._sessions: dict[str, PaperSession] = {}

    def create_session(self) -> str:
        """Create a new paper session with a unique ID.

        Returns:
            The session ID for the newly created session.
        """
        session_id = str(uuid.uuid4())
        session = PaperSession(session_id=session_id)
        self._sessions[session_id] = session
        return session_id

    def get_session(self, session_id: str) -> PaperSession:
        """Retrieve an existing session by its ID.

        Args:
            session_id: The unique identifier of the session.

        Returns:
            The PaperSession associated with the given ID.

        Raises:
            KeyError: If no session exists with the given ID.
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> None:
        """Delete a session by its ID.

        Args:
            session_id: The unique identifier of the session to delete.

        Raises:
            KeyError: If no session exists with the given ID.
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        del self._sessions[session_id]
