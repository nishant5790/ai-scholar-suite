"""Unit tests for the FastAPI server."""

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from src.api.server import create_app
from src.core.session_manager import SessionManager
from src.models.schemas import (
    CitationMetadata,
    CitationStyle,
    PaperOutline,
    OutlineSection,
    PaperState,
    SectionContent,
    SectionType,
)


@pytest.fixture
def session_manager():
    return SessionManager()


@pytest.fixture
def client(session_manager):
    app = create_app(session_manager=session_manager)
    return TestClient(app)


@pytest.fixture
def session_id(client):
    """Create a session and return its ID."""
    resp = client.post("/api/v1/sessions")
    return resp.json()["session_id"]


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------


class TestCreateSession:
    def test_returns_201(self, client):
        resp = client.post("/api/v1/sessions")
        assert resp.status_code == 201

    def test_returns_session_id(self, client):
        resp = client.post("/api/v1/sessions")
        data = resp.json()
        assert "session_id" in data
        assert isinstance(data["session_id"], str)
        assert len(data["session_id"]) > 0

    def test_multiple_sessions_unique(self, client):
        ids = {client.post("/api/v1/sessions").json()["session_id"] for _ in range(5)}
        assert len(ids) == 5


# ---------------------------------------------------------------------------
# Chat endpoint (placeholder)
# ---------------------------------------------------------------------------


class TestChat:
    def test_chat_returns_response(self, client, session_id):
        resp = client.post(
            f"/api/v1/sessions/{session_id}/chat",
            json={"message": "Hello"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "Hello" in data["response"]

    def test_chat_missing_message(self, client, session_id):
        resp = client.post(
            f"/api/v1/sessions/{session_id}/chat",
            json={},
        )
        assert resp.status_code == 422  # FastAPI validation

    def test_chat_empty_message(self, client, session_id):
        resp = client.post(
            f"/api/v1/sessions/{session_id}/chat",
            json={"message": ""},
        )
        assert resp.status_code == 422

    def test_chat_nonexistent_session(self, client):
        resp = client.post(
            "/api/v1/sessions/nonexistent/chat",
            json={"message": "Hello"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Outline endpoint
# ---------------------------------------------------------------------------


class TestOutline:
    def test_outline_no_agent_returns_500(self, client, session_id):
        resp = client.post(
            f"/api/v1/sessions/{session_id}/outline",
            json={"topic": "Machine Learning"},
        )
        assert resp.status_code == 500

    def test_outline_missing_topic(self, client, session_id):
        resp = client.post(
            f"/api/v1/sessions/{session_id}/outline",
            json={},
        )
        assert resp.status_code == 422

    def test_outline_empty_topic(self, client, session_id):
        resp = client.post(
            f"/api/v1/sessions/{session_id}/outline",
            json={"topic": ""},
        )
        assert resp.status_code == 422

    def test_outline_nonexistent_session(self, client):
        resp = client.post(
            "/api/v1/sessions/nonexistent/outline",
            json={"topic": "AI"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Section endpoint
# ---------------------------------------------------------------------------


class TestSection:
    def test_invalid_section_type(self, client, session_id):
        resp = client.post(
            f"/api/v1/sessions/{session_id}/sections/invalid_type",
            json={},
        )
        assert resp.status_code == 400
        assert "Invalid section type" in resp.json()["detail"]

    def test_valid_section_no_agent(self, client, session_id):
        resp = client.post(
            f"/api/v1/sessions/{session_id}/sections/abstract",
            json={},
        )
        assert resp.status_code == 500

    def test_section_nonexistent_session(self, client):
        resp = client.post(
            "/api/v1/sessions/nonexistent/sections/abstract",
            json={},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Reference ingestion endpoint
# ---------------------------------------------------------------------------


class TestIngestReferences:
    def test_ingest_valid_empty_folder(self, client, session_id):
        with tempfile.TemporaryDirectory() as tmpdir:
            resp = client.post(
                f"/api/v1/sessions/{session_id}/references/ingest",
                json={"folder_path": tmpdir},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["files_processed"] == 0
            assert data["files_skipped"] == 0

    def test_ingest_nonexistent_folder(self, client, session_id):
        resp = client.post(
            f"/api/v1/sessions/{session_id}/references/ingest",
            json={"folder_path": "/nonexistent/path/xyz"},
        )
        assert resp.status_code == 400

    def test_ingest_missing_folder_path(self, client, session_id):
        resp = client.post(
            f"/api/v1/sessions/{session_id}/references/ingest",
            json={},
        )
        assert resp.status_code == 422

    def test_ingest_with_txt_file(self, client, session_id):
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = os.path.join(tmpdir, "ref.txt")
            with open(txt_path, "w") as f:
                f.write("Some reference content for testing.")
            resp = client.post(
                f"/api/v1/sessions/{session_id}/references/ingest",
                json={"folder_path": tmpdir},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["files_processed"] == 1

    def test_ingest_nonexistent_session(self, client):
        resp = client.post(
            "/api/v1/sessions/nonexistent/references/ingest",
            json={"folder_path": "/tmp"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Bibliography endpoint
# ---------------------------------------------------------------------------


class TestBibliography:
    def test_empty_bibliography(self, client, session_id):
        resp = client.get(f"/api/v1/sessions/{session_id}/bibliography")
        assert resp.status_code == 200
        data = resp.json()
        assert data["bibliography"] == ""
        assert data["style"] == "apa"

    def test_bibliography_with_citations(self, client, session_id, session_manager):
        session = session_manager.get_session(session_id)
        session.paper_state.citations = {
            "ref1": CitationMetadata(
                citation_id="ref1",
                author="Smith, J.",
                title="Test Paper",
                year=2023,
                source="Journal of Testing",
            )
        }
        resp = client.get(f"/api/v1/sessions/{session_id}/bibliography")
        assert resp.status_code == 200
        data = resp.json()
        assert "Smith" in data["bibliography"]
        assert "Test Paper" in data["bibliography"]

    def test_bibliography_with_style(self, client, session_id, session_manager):
        session = session_manager.get_session(session_id)
        session.paper_state.citations = {
            "ref1": CitationMetadata(
                citation_id="ref1",
                author="Doe, A.",
                title="Another Paper",
                year=2024,
                source="Science Journal",
            )
        }
        resp = client.get(f"/api/v1/sessions/{session_id}/bibliography?style=ieee")
        assert resp.status_code == 200
        data = resp.json()
        assert data["style"] == "ieee"
        assert "Doe" in data["bibliography"]

    def test_bibliography_invalid_style(self, client, session_id):
        resp = client.get(f"/api/v1/sessions/{session_id}/bibliography?style=invalid")
        assert resp.status_code == 400

    def test_bibliography_nonexistent_session(self, client):
        resp = client.get("/api/v1/sessions/nonexistent/bibliography")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PDF export endpoint
# ---------------------------------------------------------------------------


class TestExportPdf:
    def _make_complete_state(self) -> PaperState:
        """Build a PaperState with all required sections."""
        state = PaperState(title="Test Paper", author="Test Author", topic="Testing")
        for st in SectionType:
            state.sections[st.value] = SectionContent(
                section_type=st,
                title=st.value.replace("_", " ").title(),
                content=f"Content for {st.value}.",
            )
        return state

    def test_export_pdf_success(self, client, session_id, session_manager):
        session = session_manager.get_session(session_id)
        session.paper_state = self._make_complete_state()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            out_path = f.name
        try:
            resp = client.post(
                f"/api/v1/sessions/{session_id}/export/pdf",
                json={"output_path": out_path},
            )
            assert resp.status_code == 200
            assert resp.json()["output_path"] == out_path
            assert os.path.exists(out_path)
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_export_pdf_missing_sections(self, client, session_id):
        resp = client.post(
            f"/api/v1/sessions/{session_id}/export/pdf",
            json={"output_path": "/tmp/test.pdf"},
        )
        assert resp.status_code == 400

    def test_export_pdf_missing_output_path(self, client, session_id):
        resp = client.post(
            f"/api/v1/sessions/{session_id}/export/pdf",
            json={},
        )
        assert resp.status_code == 422

    def test_export_pdf_nonexistent_session(self, client):
        resp = client.post(
            "/api/v1/sessions/nonexistent/export/pdf",
            json={"output_path": "/tmp/test.pdf"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Save / Load endpoints
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_state(self, client, session_id, session_manager):
        session = session_manager.get_session(session_id)
        session.paper_state.title = "My Paper"
        session.paper_state.topic = "AI"
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            file_path = f.name
        try:
            resp = client.post(
                f"/api/v1/sessions/{session_id}/save",
                json={"file_path": file_path},
            )
            assert resp.status_code == 200
            assert "saved" in resp.json()["message"].lower()
            assert os.path.exists(file_path)
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)

    def test_load_state(self, client, session_id, session_manager):
        # First save a state
        state = PaperState(title="Loaded Paper", topic="Testing")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write(state.model_dump_json(indent=2))
            file_path = f.name
        try:
            resp = client.post(
                f"/api/v1/sessions/{session_id}/load",
                json={"file_path": file_path},
            )
            assert resp.status_code == 200
            assert "loaded" in resp.json()["message"].lower()

            # Verify the state was actually loaded
            session = session_manager.get_session(session_id)
            assert session.paper_state.title == "Loaded Paper"
            assert session.paper_state.topic == "Testing"
        finally:
            os.unlink(file_path)

    def test_load_nonexistent_file(self, client, session_id):
        resp = client.post(
            f"/api/v1/sessions/{session_id}/load",
            json={"file_path": "/nonexistent/file.json"},
        )
        assert resp.status_code == 400

    def test_save_missing_file_path(self, client, session_id):
        resp = client.post(
            f"/api/v1/sessions/{session_id}/save",
            json={},
        )
        assert resp.status_code == 422

    def test_load_missing_file_path(self, client, session_id):
        resp = client.post(
            f"/api/v1/sessions/{session_id}/load",
            json={},
        )
        assert resp.status_code == 422

    def test_save_nonexistent_session(self, client):
        resp = client.post(
            "/api/v1/sessions/nonexistent/save",
            json={"file_path": "/tmp/test.json"},
        )
        assert resp.status_code == 404

    def test_load_nonexistent_session(self, client):
        resp = client.post(
            "/api/v1/sessions/nonexistent/load",
            json={"file_path": "/tmp/test.json"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# JSON response format
# ---------------------------------------------------------------------------


class TestResponseFormat:
    def test_all_endpoints_return_json(self, client, session_id):
        """Verify that all successful responses have JSON content type."""
        resp = client.post("/api/v1/sessions")
        assert resp.headers["content-type"] == "application/json"

        resp = client.post(
            f"/api/v1/sessions/{session_id}/chat",
            json={"message": "test"},
        )
        assert resp.headers["content-type"] == "application/json"

        resp = client.get(f"/api/v1/sessions/{session_id}/bibliography")
        assert resp.headers["content-type"] == "application/json"
