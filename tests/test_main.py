"""Tests for the application entry point."""

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestBuildApp:
    """Tests for build_app and the module-level app."""

    def test_app_is_fastapi_instance(self):
        from src.main import app

        assert isinstance(app, FastAPI)

    def test_app_has_cors_middleware(self):
        from src.main import app

        middleware_classes = [type(m).__name__ for m in app.user_middleware]
        # CORSMiddleware is added via add_middleware which stores it in user_middleware
        assert any("CORS" in name for name in middleware_classes) or any(
            "CORSMiddleware" in str(m) for m in app.user_middleware
        )

    def test_app_state_has_chroma_client(self):
        from src.main import app

        assert hasattr(app.state, "chroma_client")
        assert app.state.chroma_client is not None

    def test_app_state_has_chroma_collection(self):
        from src.main import app

        assert hasattr(app.state, "chroma_collection")
        assert app.state.chroma_collection is not None

    def test_app_state_has_settings(self):
        from src.main import app

        assert hasattr(app.state, "settings")
        assert app.state.settings is not None

    def test_session_endpoint_works(self):
        """Verify the wired SessionManager is functional via the API."""
        from src.main import app

        client = TestClient(app)
        response = client.post("/api/v1/sessions")
        assert response.status_code == 201
        assert "session_id" in response.json()


class TestMain:
    """Tests for the main() entry function."""

    def test_main_calls_uvicorn_run(self):
        with patch("src.main.uvicorn.run") as mock_run:
            from src.main import main

            main()
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args
            assert call_kwargs[1]["host"] == "0.0.0.0"
            assert call_kwargs[1]["port"] == 8000
            assert call_kwargs[0][0] == "src.main:app"
