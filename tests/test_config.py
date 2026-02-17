"""Tests for application configuration."""

import os

from src.config import Settings, get_settings


class TestSettings:
    """Tests for the Settings class."""

    def test_default_values(self):
        settings = Settings(openai_api_key="test-key")
        assert settings.llm_model == "gpt-4o-mini"
        assert settings.chromadb_path == "./chroma_data"
        assert settings.output_dir == "./output"
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000

    def test_openai_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
        settings = Settings()
        assert settings.openai_api_key == "sk-test-123"

    def test_env_override_llm_model(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        settings = Settings()
        assert settings.llm_model == "gpt-4o"

    def test_env_override_chromadb_path(self, monkeypatch):
        monkeypatch.setenv("CHROMADB_PATH", "/data/chroma")
        settings = Settings()
        assert settings.chromadb_path == "/data/chroma"

    def test_env_override_output_dir(self, monkeypatch):
        monkeypatch.setenv("OUTPUT_DIR", "/data/output")
        settings = Settings()
        assert settings.output_dir == "/data/output"

    def test_env_override_api_host(self, monkeypatch):
        monkeypatch.setenv("API_HOST", "127.0.0.1")
        settings = Settings()
        assert settings.api_host == "127.0.0.1"

    def test_env_override_api_port(self, monkeypatch):
        monkeypatch.setenv("API_PORT", "9000")
        settings = Settings()
        assert settings.api_port == 9000

    def test_get_settings_returns_settings_instance(self):
        settings = get_settings()
        assert isinstance(settings, Settings)
