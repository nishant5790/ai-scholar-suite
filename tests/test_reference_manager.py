"""Unit tests for the ReferenceManagerTool."""

import json

import pytest

from src.models.schemas import CitationMetadata, CitationStyle
from src.tools.reference_manager import ReferenceManagerTool


@pytest.fixture
def ref_manager():
    """Create a fresh ReferenceManagerTool instance."""
    return ReferenceManagerTool()


@pytest.fixture
def sample_citation():
    """Create a sample CitationMetadata."""
    return CitationMetadata(
        citation_id="cite_1",
        author="Smith, J.",
        title="Deep Learning Advances",
        year=2023,
        source="Journal of AI Research",
    )


@pytest.fixture
def sample_citation_2():
    """Create a second sample CitationMetadata."""
    return CitationMetadata(
        citation_id="cite_2",
        author="Doe, A.",
        title="Neural Network Optimization",
        year=2022,
        source="Machine Learning Conference",
        doi="10.1234/example",
    )


class TestAddCitation:
    def test_add_single_citation(self, ref_manager, sample_citation):
        cid = ref_manager.add_citation(sample_citation)
        assert cid == "cite_1"
        assert "cite_1" in ref_manager.citations
        assert ref_manager.citations["cite_1"].author == "Smith, J."

    def test_add_multiple_citations(self, ref_manager, sample_citation, sample_citation_2):
        cid1 = ref_manager.add_citation(sample_citation)
        cid2 = ref_manager.add_citation(sample_citation_2)
        assert cid1 == "cite_1"
        assert cid2 == "cite_2"
        assert len(ref_manager.citations) == 2

    def test_stored_metadata_matches_input(self, ref_manager, sample_citation):
        ref_manager.add_citation(sample_citation)
        stored = ref_manager.citations["cite_1"]
        assert stored.author == sample_citation.author
        assert stored.title == sample_citation.title
        assert stored.year == sample_citation.year
        assert stored.source == sample_citation.source


class TestDeduplication:
    def test_duplicate_returns_same_id(self, ref_manager):
        meta1 = CitationMetadata(
            citation_id="cite_1", author="Smith, J.", title="Deep Learning", year=2023, source="Journal A"
        )
        meta2 = CitationMetadata(
            citation_id="cite_2", author="Smith, J.", title="Deep Learning", year=2023, source="Journal B"
        )
        cid1 = ref_manager.add_citation(meta1)
        cid2 = ref_manager.add_citation(meta2)
        assert cid1 == cid2 == "cite_1"

    def test_duplicate_does_not_increase_count(self, ref_manager):
        meta1 = CitationMetadata(
            citation_id="cite_1", author="Doe, A.", title="ML Paper", year=2022, source="Conf A"
        )
        meta2 = CitationMetadata(
            citation_id="cite_2", author="Doe, A.", title="ML Paper", year=2022, source="Conf B"
        )
        ref_manager.add_citation(meta1)
        ref_manager.add_citation(meta2)
        assert len(ref_manager.citations) == 1

    def test_different_author_not_duplicate(self, ref_manager):
        meta1 = CitationMetadata(
            citation_id="cite_1", author="Smith, J.", title="Paper", year=2023, source="S"
        )
        meta2 = CitationMetadata(
            citation_id="cite_2", author="Doe, A.", title="Paper", year=2023, source="S"
        )
        ref_manager.add_citation(meta1)
        ref_manager.add_citation(meta2)
        assert len(ref_manager.citations) == 2

    def test_different_year_not_duplicate(self, ref_manager):
        meta1 = CitationMetadata(
            citation_id="cite_1", author="Smith, J.", title="Paper", year=2022, source="S"
        )
        meta2 = CitationMetadata(
            citation_id="cite_2", author="Smith, J.", title="Paper", year=2023, source="S"
        )
        ref_manager.add_citation(meta1)
        ref_manager.add_citation(meta2)
        assert len(ref_manager.citations) == 2

    def test_different_title_not_duplicate(self, ref_manager):
        meta1 = CitationMetadata(
            citation_id="cite_1", author="Smith, J.", title="Paper A", year=2023, source="S"
        )
        meta2 = CitationMetadata(
            citation_id="cite_2", author="Smith, J.", title="Paper B", year=2023, source="S"
        )
        ref_manager.add_citation(meta1)
        ref_manager.add_citation(meta2)
        assert len(ref_manager.citations) == 2


class TestGenerateBibliography:
    def test_empty_bibliography(self, ref_manager):
        assert ref_manager.generate_bibliography() == ""

    def test_apa_format(self, ref_manager, sample_citation):
        ref_manager.add_citation(sample_citation)
        bib = ref_manager.generate_bibliography(CitationStyle.APA)
        assert bib == "Smith, J. (2023). Deep Learning Advances. Journal of AI Research."

    def test_ieee_format(self, ref_manager, sample_citation):
        ref_manager.add_citation(sample_citation)
        bib = ref_manager.generate_bibliography(CitationStyle.IEEE)
        assert bib == '[1] Smith, J., "Deep Learning Advances," Journal of AI Research, 2023.'

    def test_mla_format(self, ref_manager, sample_citation):
        ref_manager.add_citation(sample_citation)
        bib = ref_manager.generate_bibliography(CitationStyle.MLA)
        assert bib == 'Smith, J. "Deep Learning Advances." Journal of AI Research, 2023.'

    def test_default_style_is_apa(self, ref_manager, sample_citation):
        ref_manager.add_citation(sample_citation)
        bib_default = ref_manager.generate_bibliography()
        bib_apa = ref_manager.generate_bibliography(CitationStyle.APA)
        assert bib_default == bib_apa

    def test_multiple_entries(self, ref_manager, sample_citation, sample_citation_2):
        ref_manager.add_citation(sample_citation)
        ref_manager.add_citation(sample_citation_2)
        bib = ref_manager.generate_bibliography(CitationStyle.APA)
        lines = bib.strip().split("\n")
        assert len(lines) == 2
        assert "Smith, J." in lines[0]
        assert "Doe, A." in lines[1]

    def test_bibliography_contains_all_authors_and_titles(self, ref_manager, sample_citation, sample_citation_2):
        ref_manager.add_citation(sample_citation)
        ref_manager.add_citation(sample_citation_2)
        for style in CitationStyle:
            bib = ref_manager.generate_bibliography(style)
            assert "Smith, J." in bib
            assert "Doe, A." in bib
            assert "Deep Learning Advances" in bib
            assert "Neural Network Optimization" in bib


class TestGetInlineMarker:
    def test_apa_marker(self, ref_manager, sample_citation):
        ref_manager.citation_style = CitationStyle.APA
        ref_manager.add_citation(sample_citation)
        marker = ref_manager.get_inline_marker("cite_1")
        assert marker == "(Smith, J., 2023)"

    def test_ieee_marker(self, ref_manager, sample_citation):
        ref_manager.citation_style = CitationStyle.IEEE
        ref_manager.add_citation(sample_citation)
        marker = ref_manager.get_inline_marker("cite_1")
        assert marker == "[1]"

    def test_mla_marker(self, ref_manager, sample_citation):
        ref_manager.citation_style = CitationStyle.MLA
        ref_manager.add_citation(sample_citation)
        marker = ref_manager.get_inline_marker("cite_1")
        assert marker == "(Smith, J. 2023)"

    def test_ieee_numbering_order(self, ref_manager, sample_citation, sample_citation_2):
        ref_manager.citation_style = CitationStyle.IEEE
        ref_manager.add_citation(sample_citation)
        ref_manager.add_citation(sample_citation_2)
        assert ref_manager.get_inline_marker("cite_1") == "[1]"
        assert ref_manager.get_inline_marker("cite_2") == "[2]"

    def test_unknown_citation_id_raises(self, ref_manager):
        with pytest.raises(ValueError, match="Citation ID not found"):
            ref_manager.get_inline_marker("nonexistent")


class TestRunMethod:
    def test_run_add_action(self, ref_manager):
        result = ref_manager._run(
            action="add",
            metadata={
                "citation_id": "cite_1",
                "author": "Smith, J.",
                "title": "Paper",
                "year": 2023,
                "source": "Journal",
            },
        )
        data = json.loads(result)
        assert data["citation_id"] == "cite_1"

    def test_run_bibliography_action(self, ref_manager, sample_citation):
        ref_manager.add_citation(sample_citation)
        result = ref_manager._run(action="bibliography", style="apa")
        assert "Smith, J." in result

    def test_run_bibliography_default_style(self, ref_manager, sample_citation):
        ref_manager.add_citation(sample_citation)
        result = ref_manager._run(action="bibliography")
        assert "Smith, J. (2023)" in result

    def test_run_marker_action(self, ref_manager, sample_citation):
        ref_manager.add_citation(sample_citation)
        result = ref_manager._run(action="marker", citation_id="cite_1")
        assert result == "(Smith, J., 2023)"

    def test_run_add_missing_metadata(self, ref_manager):
        result = ref_manager._run(action="add")
        data = json.loads(result)
        assert "error" in data

    def test_run_marker_missing_id(self, ref_manager):
        result = ref_manager._run(action="marker")
        data = json.loads(result)
        assert "error" in data

    def test_run_unknown_action(self, ref_manager):
        result = ref_manager._run(action="unknown")
        data = json.loads(result)
        assert "error" in data
