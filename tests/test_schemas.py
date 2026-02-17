"""Unit tests for data models in src/models/schemas.py."""

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    CitationMetadata,
    CitationStyle,
    ErrorResponse,
    IngestionResult,
    OutlineSection,
    PaperOutline,
    PaperState,
    SectionContent,
    SectionType,
)


class TestSectionType:
    def test_all_standard_sections_exist(self):
        expected = {
            "abstract",
            "introduction",
            "literature_review",
            "methodology",
            "results",
            "discussion",
            "conclusion",
        }
        assert {s.value for s in SectionType} == expected

    def test_section_type_is_string_enum(self):
        assert SectionType.ABSTRACT == "abstract"
        assert isinstance(SectionType.ABSTRACT, str)


class TestOutlineSection:
    def test_basic_section(self):
        section = OutlineSection(
            section_type=SectionType.INTRODUCTION,
            title="Introduction",
            key_points=["Background", "Motivation"],
        )
        assert section.section_type == SectionType.INTRODUCTION
        assert section.title == "Introduction"
        assert section.key_points == ["Background", "Motivation"]
        assert section.subsections == []

    def test_nested_subsections(self):
        child = OutlineSection(
            section_type=SectionType.METHODOLOGY,
            title="Data Collection",
            key_points=["Surveys"],
        )
        parent = OutlineSection(
            section_type=SectionType.METHODOLOGY,
            title="Methodology",
            key_points=["Overview"],
            subsections=[child],
        )
        assert len(parent.subsections) == 1
        assert parent.subsections[0].title == "Data Collection"

    def test_missing_required_fields_raises(self):
        with pytest.raises(ValidationError):
            OutlineSection(section_type=SectionType.ABSTRACT, title="Abstract")


class TestPaperOutline:
    def test_outline_with_sections(self):
        sections = [
            OutlineSection(
                section_type=SectionType.ABSTRACT,
                title="Abstract",
                key_points=["Summary"],
            )
        ]
        outline = PaperOutline(topic="AI in Healthcare", sections=sections)
        assert outline.topic == "AI in Healthcare"
        assert len(outline.sections) == 1


class TestSectionContent:
    def test_section_content_defaults(self):
        sc = SectionContent(
            section_type=SectionType.RESULTS,
            title="Results",
            content="Our findings show...",
        )
        assert sc.citations == []

    def test_section_content_with_citations(self):
        sc = SectionContent(
            section_type=SectionType.DISCUSSION,
            title="Discussion",
            content="As noted by...",
            citations=["cite1", "cite2"],
        )
        assert sc.citations == ["cite1", "cite2"]


class TestCitationMetadata:
    def test_citation_with_doi(self):
        c = CitationMetadata(
            citation_id="c1",
            author="Smith, J.",
            title="A Study",
            year=2023,
            source="Nature",
            doi="10.1234/example",
        )
        assert c.doi == "10.1234/example"

    def test_citation_without_doi(self):
        c = CitationMetadata(
            citation_id="c2",
            author="Doe, A.",
            title="Another Study",
            year=2022,
            source="Science",
        )
        assert c.doi is None


class TestCitationStyle:
    def test_all_styles(self):
        assert {s.value for s in CitationStyle} == {"apa", "ieee", "mla"}


class TestPaperState:
    def test_default_state(self):
        state = PaperState()
        assert state.title == ""
        assert state.author == ""
        assert state.topic == ""
        assert state.outline is None
        assert state.sections == {}
        assert state.citations == {}
        assert state.citation_style == CitationStyle.APA

    def test_full_state(self):
        outline = PaperOutline(
            topic="ML",
            sections=[
                OutlineSection(
                    section_type=SectionType.ABSTRACT,
                    title="Abstract",
                    key_points=["Summary"],
                )
            ],
        )
        section = SectionContent(
            section_type=SectionType.ABSTRACT,
            title="Abstract",
            content="This paper...",
        )
        citation = CitationMetadata(
            citation_id="c1",
            author="Smith",
            title="Paper",
            year=2023,
            source="Journal",
        )
        state = PaperState(
            title="My Paper",
            author="Author",
            topic="ML",
            outline=outline,
            sections={"abstract": section},
            citations={"c1": citation},
            citation_style=CitationStyle.IEEE,
        )
        assert state.title == "My Paper"
        assert state.citation_style == CitationStyle.IEEE
        assert "abstract" in state.sections
        assert "c1" in state.citations


class TestIngestionResult:
    def test_ingestion_result(self):
        result = IngestionResult(
            files_processed=5,
            files_skipped=2,
            skipped_files=["image.png", "data.csv"],
            total_chunks=42,
        )
        assert result.files_processed == 5
        assert result.files_skipped == 2
        assert len(result.skipped_files) == 2
        assert result.total_chunks == 42


class TestErrorResponse:
    def test_error_response_defaults(self):
        err = ErrorResponse(error="validation_error", message="Invalid input")
        assert err.details == {}

    def test_error_response_with_details(self):
        err = ErrorResponse(
            error="missing_section",
            message="Sections missing",
            details={"missing": ["abstract", "conclusion"]},
        )
        assert err.details["missing"] == ["abstract", "conclusion"]
