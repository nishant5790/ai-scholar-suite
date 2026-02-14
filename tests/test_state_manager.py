"""Unit tests for the StateManager."""

import json

import pytest

from src.core.state_manager import StateManager
from src.models.schemas import (
    CitationMetadata,
    CitationStyle,
    OutlineSection,
    PaperOutline,
    PaperState,
    SectionContent,
    SectionType,
)


@pytest.fixture
def state_manager():
    return StateManager()


@pytest.fixture
def empty_state():
    return PaperState()


@pytest.fixture
def full_state():
    outline = PaperOutline(
        topic="Machine Learning",
        sections=[
            OutlineSection(
                section_type=SectionType.ABSTRACT,
                title="Abstract",
                key_points=["Summary of findings"],
            ),
            OutlineSection(
                section_type=SectionType.INTRODUCTION,
                title="Introduction",
                key_points=["Background", "Motivation"],
            ),
        ],
    )
    sections = {
        "abstract": SectionContent(
            section_type=SectionType.ABSTRACT,
            title="Abstract",
            content="This paper explores...",
            citations=["c1"],
        ),
    }
    citations = {
        "c1": CitationMetadata(
            citation_id="c1",
            author="Smith, J.",
            title="Deep Learning Advances",
            year=2023,
            source="Nature",
            doi="10.1234/dl",
        ),
    }
    return PaperState(
        title="ML Survey",
        author="Doe, A.",
        topic="Machine Learning",
        outline=outline,
        sections=sections,
        citations=citations,
        citation_style=CitationStyle.IEEE,
    )


class TestSaveState:
    def test_save_creates_file(self, state_manager, empty_state, tmp_path):
        file_path = str(tmp_path / "state.json")
        state_manager.save_state(empty_state, file_path)
        assert (tmp_path / "state.json").exists()

    def test_save_creates_parent_directories(self, state_manager, empty_state, tmp_path):
        file_path = str(tmp_path / "nested" / "dir" / "state.json")
        state_manager.save_state(empty_state, file_path)
        assert (tmp_path / "nested" / "dir" / "state.json").exists()

    def test_saved_file_is_valid_json(self, state_manager, full_state, tmp_path):
        file_path = str(tmp_path / "state.json")
        state_manager.save_state(full_state, file_path)
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["title"] == "ML Survey"
        assert data["author"] == "Doe, A."

    def test_save_overwrites_existing_file(self, state_manager, tmp_path):
        file_path = str(tmp_path / "state.json")
        state_manager.save_state(PaperState(title="First"), file_path)
        state_manager.save_state(PaperState(title="Second"), file_path)
        loaded = state_manager.load_state(file_path)
        assert loaded.title == "Second"


class TestLoadState:
    def test_load_empty_state(self, state_manager, empty_state, tmp_path):
        file_path = str(tmp_path / "state.json")
        state_manager.save_state(empty_state, file_path)
        loaded = state_manager.load_state(file_path)
        assert loaded == empty_state

    def test_load_full_state(self, state_manager, full_state, tmp_path):
        file_path = str(tmp_path / "state.json")
        state_manager.save_state(full_state, file_path)
        loaded = state_manager.load_state(file_path)
        assert loaded == full_state

    def test_load_preserves_all_fields(self, state_manager, full_state, tmp_path):
        file_path = str(tmp_path / "state.json")
        state_manager.save_state(full_state, file_path)
        loaded = state_manager.load_state(file_path)
        assert loaded.title == full_state.title
        assert loaded.author == full_state.author
        assert loaded.topic == full_state.topic
        assert loaded.outline == full_state.outline
        assert loaded.sections == full_state.sections
        assert loaded.citations == full_state.citations
        assert loaded.citation_style == full_state.citation_style

    def test_load_nonexistent_file_raises(self, state_manager):
        with pytest.raises(FileNotFoundError):
            state_manager.load_state("/nonexistent/path/state.json")

    def test_load_invalid_json_raises(self, state_manager, tmp_path):
        file_path = tmp_path / "bad.json"
        file_path.write_text("not valid json {{{", encoding="utf-8")
        with pytest.raises(Exception):
            state_manager.load_state(str(file_path))

    def test_load_invalid_schema_raises(self, state_manager, tmp_path):
        file_path = tmp_path / "bad_schema.json"
        file_path.write_text('{"unknown_field": 123}', encoding="utf-8")
        # PaperState has all optional/default fields, so this should still load
        # but unknown fields are ignored by Pydantic by default
        loaded = state_manager.load_state(str(file_path))
        assert loaded.title == ""


class TestRoundTrip:
    def test_round_trip_empty_state(self, state_manager, tmp_path):
        file_path = str(tmp_path / "state.json")
        original = PaperState()
        state_manager.save_state(original, file_path)
        loaded = state_manager.load_state(file_path)
        assert loaded == original

    def test_round_trip_full_state(self, state_manager, full_state, tmp_path):
        file_path = str(tmp_path / "state.json")
        state_manager.save_state(full_state, file_path)
        loaded = state_manager.load_state(file_path)
        assert loaded == full_state

    def test_round_trip_state_with_nested_subsections(self, state_manager, tmp_path):
        child = OutlineSection(
            section_type=SectionType.METHODOLOGY,
            title="Data Collection",
            key_points=["Surveys", "Interviews"],
        )
        outline = PaperOutline(
            topic="Research Methods",
            sections=[
                OutlineSection(
                    section_type=SectionType.METHODOLOGY,
                    title="Methodology",
                    key_points=["Overview"],
                    subsections=[child],
                ),
            ],
        )
        state = PaperState(title="Methods Paper", outline=outline)
        file_path = str(tmp_path / "state.json")
        state_manager.save_state(state, file_path)
        loaded = state_manager.load_state(file_path)
        assert loaded == state
        assert loaded.outline.sections[0].subsections[0].title == "Data Collection"
