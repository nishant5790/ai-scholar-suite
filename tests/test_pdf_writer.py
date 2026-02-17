"""Unit tests for the PDFWriterTool."""

import os
import tempfile

import pytest

from src.models.schemas import (
    CitationMetadata,
    CitationStyle,
    PaperState,
    SectionContent,
    SectionType,
)
from src.tools.pdf_writer import PDFWriterTool, REQUIRED_SECTIONS
from src.tools.reference_manager import ReferenceManagerTool


def _make_section(section_type: SectionType, content: str = "Sample content.") -> SectionContent:
    """Helper to create a SectionContent for a given type."""
    return SectionContent(
        section_type=section_type,
        title=section_type.value.replace("_", " ").title(),
        content=content,
    )


def _make_complete_state(**overrides) -> PaperState:
    """Create a PaperState with all required sections populated."""
    sections = {}
    for st in REQUIRED_SECTIONS:
        sections[st.value] = _make_section(st, f"Content for {st.value} section.")
    defaults = dict(
        title="Test Research Paper",
        author="Jane Doe",
        topic="AI Testing",
        sections=sections,
    )
    defaults.update(overrides)
    return PaperState(**defaults)


@pytest.fixture
def complete_state():
    return _make_complete_state()


@pytest.fixture
def pdf_tool(complete_state):
    return PDFWriterTool(paper_state=complete_state)


@pytest.fixture
def tmp_pdf(tmp_path):
    return str(tmp_path / "output.pdf")


class TestMissingSections:
    def test_all_sections_missing(self, tmp_pdf):
        tool = PDFWriterTool(paper_state=PaperState())
        result = tool._run(tmp_pdf)
        assert result.startswith("Error: Missing required sections:")
        for st in REQUIRED_SECTIONS:
            assert st.value in result

    def test_one_section_missing(self, tmp_pdf):
        state = _make_complete_state()
        del state.sections[SectionType.METHODOLOGY.value]
        tool = PDFWriterTool(paper_state=state)
        result = tool._run(tmp_pdf)
        assert "methodology" in result
        assert result.startswith("Error:")

    def test_multiple_sections_missing(self, tmp_pdf):
        state = _make_complete_state()
        del state.sections[SectionType.ABSTRACT.value]
        del state.sections[SectionType.CONCLUSION.value]
        tool = PDFWriterTool(paper_state=state)
        result = tool._run(tmp_pdf)
        assert "abstract" in result
        assert "conclusion" in result

    def test_no_missing_sections_succeeds(self, pdf_tool, tmp_pdf):
        result = pdf_tool._run(tmp_pdf)
        assert result == tmp_pdf
        assert os.path.exists(tmp_pdf)


class TestPDFGeneration:
    def test_creates_valid_pdf_file(self, pdf_tool, tmp_pdf):
        pdf_tool._run(tmp_pdf)
        with open(tmp_pdf, "rb") as f:
            header = f.read(5)
        assert header == b"%PDF-"

    def test_returns_output_path(self, pdf_tool, tmp_pdf):
        result = pdf_tool._run(tmp_pdf)
        assert result == tmp_pdf

    def test_pdf_file_not_empty(self, pdf_tool, tmp_pdf):
        pdf_tool._run(tmp_pdf)
        assert os.path.getsize(tmp_pdf) > 0

    def test_pdf_with_nested_directory(self, tmp_path):
        nested = tmp_path / "sub" / "dir"
        nested.mkdir(parents=True)
        out = str(nested / "paper.pdf")
        tool = PDFWriterTool(paper_state=_make_complete_state())
        result = tool._run(out)
        assert result == out
        assert os.path.exists(out)


class TestTitlePage:
    def test_pdf_contains_title(self, tmp_pdf):
        state = _make_complete_state(title="Quantum Computing Survey")
        tool = PDFWriterTool(paper_state=state)
        tool._run(tmp_pdf)
        text = _extract_pdf_text(tmp_pdf)
        assert "Quantum Computing Survey" in text

    def test_pdf_contains_author(self, tmp_pdf):
        state = _make_complete_state(author="Dr. Alice Smith")
        tool = PDFWriterTool(paper_state=state)
        tool._run(tmp_pdf)
        text = _extract_pdf_text(tmp_pdf)
        assert "Dr. Alice Smith" in text

    def test_default_title_when_empty(self, tmp_pdf):
        state = _make_complete_state(title="")
        tool = PDFWriterTool(paper_state=state)
        tool._run(tmp_pdf)
        text = _extract_pdf_text(tmp_pdf)
        assert "Untitled Paper" in text

    def test_default_author_when_empty(self, tmp_pdf):
        state = _make_complete_state(author="")
        tool = PDFWriterTool(paper_state=state)
        tool._run(tmp_pdf)
        text = _extract_pdf_text(tmp_pdf)
        assert "Unknown Author" in text


class TestSectionRendering:
    def test_section_headings_in_pdf(self, pdf_tool, tmp_pdf):
        pdf_tool._run(tmp_pdf)
        text = _extract_pdf_text(tmp_pdf)
        for st in REQUIRED_SECTIONS:
            section = pdf_tool.paper_state.sections[st.value]
            assert section.title in text

    def test_section_content_in_pdf(self, tmp_pdf):
        state = _make_complete_state()
        state.sections[SectionType.ABSTRACT.value] = _make_section(
            SectionType.ABSTRACT, "This is a unique abstract paragraph."
        )
        tool = PDFWriterTool(paper_state=state)
        tool._run(tmp_pdf)
        text = _extract_pdf_text(tmp_pdf)
        assert "unique abstract paragraph" in text

    def test_multiline_content(self, tmp_pdf):
        state = _make_complete_state()
        state.sections[SectionType.INTRODUCTION.value] = _make_section(
            SectionType.INTRODUCTION,
            "First paragraph of intro.\n\nSecond paragraph of intro.",
        )
        tool = PDFWriterTool(paper_state=state)
        tool._run(tmp_pdf)
        text = _extract_pdf_text(tmp_pdf)
        assert "First paragraph" in text
        assert "Second paragraph" in text


class TestBibliography:
    def test_no_bibliography_when_no_citations(self, pdf_tool, tmp_pdf):
        pdf_tool._run(tmp_pdf)
        text = _extract_pdf_text(tmp_pdf)
        assert "References" not in text

    def test_bibliography_from_reference_manager(self, tmp_pdf):
        state = _make_complete_state()
        ref_mgr = ReferenceManagerTool()
        ref_mgr.add_citation(CitationMetadata(
            citation_id="c1",
            author="Smith, J.",
            title="Deep Learning Advances",
            year=2023,
            source="Journal of AI",
        ))
        tool = PDFWriterTool(paper_state=state, reference_manager=ref_mgr)
        tool._run(tmp_pdf)
        text = _extract_pdf_text(tmp_pdf)
        assert "References" in text
        assert "Smith, J." in text
        assert "Deep Learning Advances" in text

    def test_bibliography_from_paper_state_citations(self, tmp_pdf):
        state = _make_complete_state()
        state.citations = {
            "c1": CitationMetadata(
                citation_id="c1",
                author="Doe, A.",
                title="Neural Networks",
                year=2022,
                source="ML Conference",
            )
        }
        tool = PDFWriterTool(paper_state=state)
        tool._run(tmp_pdf)
        text = _extract_pdf_text(tmp_pdf)
        assert "References" in text
        assert "Doe, A." in text
        assert "Neural Networks" in text

    def test_bibliography_multiple_entries(self, tmp_pdf):
        state = _make_complete_state()
        ref_mgr = ReferenceManagerTool()
        ref_mgr.add_citation(CitationMetadata(
            citation_id="c1", author="Smith, J.", title="Paper One", year=2023, source="Journal A",
        ))
        ref_mgr.add_citation(CitationMetadata(
            citation_id="c2", author="Doe, A.", title="Paper Two", year=2022, source="Journal B",
        ))
        tool = PDFWriterTool(paper_state=state, reference_manager=ref_mgr)
        tool._run(tmp_pdf)
        text = _extract_pdf_text(tmp_pdf)
        assert "Smith, J." in text
        assert "Doe, A." in text

    def test_bibliography_respects_citation_style(self, tmp_pdf):
        state = _make_complete_state(citation_style=CitationStyle.IEEE)
        ref_mgr = ReferenceManagerTool()
        ref_mgr.add_citation(CitationMetadata(
            citation_id="c1", author="Smith, J.", title="Paper", year=2023, source="Journal",
        ))
        tool = PDFWriterTool(paper_state=state, reference_manager=ref_mgr)
        tool._run(tmp_pdf)
        text = _extract_pdf_text(tmp_pdf)
        # IEEE format uses [N] numbering
        assert "[1]" in text


class TestGetMissingSections:
    def test_no_missing_when_complete(self, pdf_tool):
        assert pdf_tool._get_missing_sections() == []

    def test_all_missing_when_empty(self):
        tool = PDFWriterTool(paper_state=PaperState())
        missing = tool._get_missing_sections()
        assert len(missing) == len(REQUIRED_SECTIONS)

    def test_returns_correct_missing_types(self):
        state = _make_complete_state()
        del state.sections[SectionType.RESULTS.value]
        del state.sections[SectionType.DISCUSSION.value]
        tool = PDFWriterTool(paper_state=state)
        missing = tool._get_missing_sections()
        assert SectionType.RESULTS in missing
        assert SectionType.DISCUSSION in missing
        assert len(missing) == 2


def _extract_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF file using PyPDF2."""
    from PyPDF2 import PdfReader

    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text
