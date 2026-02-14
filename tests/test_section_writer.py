"""Unit tests for the SectionWriterTool."""

import json
from unittest.mock import MagicMock

import pytest

from src.models.schemas import (
    OutlineSection,
    PaperOutline,
    PaperState,
    SectionContent,
    SectionType,
)
from src.tools.section_writer import VALID_SECTION_TYPES, SectionWriterTool


def make_llm_response(response_data: dict) -> MagicMock:
    """Create a mock LLM that returns a JSON response."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps(response_data)
    mock_llm.invoke.return_value = mock_response
    return mock_llm


def make_section_response(section_type: str = "introduction", title: str = "Introduction",
                          content: str = "This paper introduces...", citations: list[str] | None = None) -> dict:
    """Create a standard section response dict."""
    return {
        "title": title,
        "content": content,
        "citations": citations or [],
    }


@pytest.fixture
def basic_llm():
    """LLM mock returning a basic introduction section."""
    return make_llm_response(make_section_response())


@pytest.fixture
def paper_state_with_outline():
    """PaperState with an outline set."""
    return PaperState(
        title="Test Paper",
        topic="Machine Learning",
        outline=PaperOutline(
            topic="Machine Learning",
            sections=[
                OutlineSection(
                    section_type=SectionType.ABSTRACT,
                    title="Abstract",
                    key_points=["Overview of ML"],
                ),
                OutlineSection(
                    section_type=SectionType.INTRODUCTION,
                    title="Introduction",
                    key_points=["Background", "Motivation"],
                ),
            ],
        ),
    )


@pytest.fixture
def paper_state_with_sections():
    """PaperState with previously generated sections."""
    return PaperState(
        title="Test Paper",
        topic="Machine Learning",
        sections={
            "abstract": SectionContent(
                section_type=SectionType.ABSTRACT,
                title="Abstract",
                content="This paper explores machine learning techniques.",
                citations=[],
            ),
        },
    )


@pytest.fixture
def tool(basic_llm):
    """SectionWriterTool with a basic mock LLM."""
    return SectionWriterTool(llm=basic_llm)


class TestSectionTypeValidation:
    """Requirement 2.5: Return error for unrecognized section types."""

    def test_valid_section_type_accepted(self, tool):
        result = tool._run(section_name="introduction")
        assert isinstance(result, SectionContent)

    def test_all_valid_types_accepted(self, basic_llm):
        for st in VALID_SECTION_TYPES:
            tool = SectionWriterTool(llm=basic_llm)
            result = tool._run(section_name=st)
            assert isinstance(result, SectionContent)

    def test_unrecognized_type_raises_error(self, tool):
        with pytest.raises(ValueError, match="Unrecognized section type"):
            tool._run(section_name="bibliography")

    def test_error_lists_valid_types(self, tool):
        with pytest.raises(ValueError) as exc_info:
            tool._run(section_name="invalid_section")
        error_msg = str(exc_info.value)
        for vt in VALID_SECTION_TYPES:
            assert vt in error_msg

    def test_case_insensitive_validation(self, tool):
        result = tool._run(section_name="INTRODUCTION")
        assert isinstance(result, SectionContent)

    def test_whitespace_trimmed(self, tool):
        result = tool._run(section_name="  introduction  ")
        assert isinstance(result, SectionContent)


class TestSectionContentGeneration:
    """Requirement 2.1: Generate academically-toned text for section type."""

    def test_returns_section_content(self, tool):
        result = tool._run(section_name="introduction")
        assert isinstance(result, SectionContent)
        assert result.section_type == SectionType.INTRODUCTION

    def test_content_has_title(self, tool):
        result = tool._run(section_name="introduction")
        assert result.title == "Introduction"

    def test_content_has_text(self, tool):
        result = tool._run(section_name="introduction")
        assert result.content == "This paper introduces..."

    def test_section_type_matches_request(self, basic_llm):
        for st in SectionType:
            resp = make_section_response(
                section_type=st.value,
                title=st.value.replace("_", " ").title(),
                content=f"Content for {st.value}",
            )
            llm = make_llm_response(resp)
            tool = SectionWriterTool(llm=llm)
            result = tool._run(section_name=st.value)
            assert result.section_type == st


class TestOutlineContext:
    """Requirement 2.2: Maintain consistency with paper outline."""

    def test_outline_included_in_prompt(self, basic_llm, paper_state_with_outline):
        tool = SectionWriterTool(llm=basic_llm, paper_state=paper_state_with_outline)
        tool._run(section_name="introduction")

        prompt = basic_llm.invoke.call_args[0][0]
        assert "Paper Outline" in prompt
        assert "Machine Learning" in prompt

    def test_no_outline_still_works(self, basic_llm):
        tool = SectionWriterTool(llm=basic_llm, paper_state=PaperState())
        result = tool._run(section_name="introduction")
        assert isinstance(result, SectionContent)

        prompt = basic_llm.invoke.call_args[0][0]
        assert "Paper Outline" not in prompt


class TestPreviousSectionsContext:
    """Requirement 2.2: Maintain consistency with previously generated sections."""

    def test_previous_sections_included_in_prompt(self, basic_llm, paper_state_with_sections):
        tool = SectionWriterTool(llm=basic_llm, paper_state=paper_state_with_sections)
        tool._run(section_name="introduction")

        prompt = basic_llm.invoke.call_args[0][0]
        assert "Previously Generated Sections" in prompt
        assert "This paper explores machine learning techniques." in prompt

    def test_no_previous_sections_still_works(self, basic_llm):
        tool = SectionWriterTool(llm=basic_llm, paper_state=PaperState())
        result = tool._run(section_name="introduction")
        assert isinstance(result, SectionContent)

        prompt = basic_llm.invoke.call_args[0][0]
        assert "Previously Generated Sections" not in prompt


class TestReferenceIntegration:
    """Requirement 2.3: Incorporate relevant citations from vector store."""

    def test_vector_store_queried(self, basic_llm):
        mock_vs = MagicMock()
        mock_vs.query.return_value = {
            "documents": [["Reference about neural networks and deep learning."]],
        }
        tool = SectionWriterTool(llm=basic_llm, vector_store=mock_vs)
        tool._run(section_name="introduction")

        mock_vs.query.assert_called_once()
        prompt = basic_llm.invoke.call_args[0][0]
        assert "Reference about neural networks and deep learning." in prompt

    def test_vector_store_query_uses_topic(self, basic_llm):
        mock_vs = MagicMock()
        mock_vs.query.return_value = {"documents": [["Some ref"]]}
        state = PaperState(topic="Quantum Computing")
        tool = SectionWriterTool(llm=basic_llm, vector_store=mock_vs, paper_state=state)
        tool._run(section_name="introduction")

        query_arg = mock_vs.query.call_args[1]["query_texts"][0]
        assert "Quantum Computing" in query_arg

    def test_no_vector_store_still_works(self, basic_llm):
        tool = SectionWriterTool(llm=basic_llm, vector_store=None)
        result = tool._run(section_name="introduction")
        assert isinstance(result, SectionContent)

    def test_vector_store_error_handled(self, basic_llm):
        mock_vs = MagicMock()
        mock_vs.query.side_effect = RuntimeError("Connection failed")
        tool = SectionWriterTool(llm=basic_llm, vector_store=mock_vs)

        result = tool._run(section_name="introduction")
        assert isinstance(result, SectionContent)

    def test_citations_returned_from_llm(self):
        resp = make_section_response(
            content="As shown by Smith (cite_1)...",
            citations=["cite_1", "cite_2"],
        )
        llm = make_llm_response(resp)
        tool = SectionWriterTool(llm=llm)
        result = tool._run(section_name="introduction")
        assert result.citations == ["cite_1", "cite_2"]


class TestFeedbackRevision:
    """Requirement 2.4: Revise section content based on feedback."""

    def test_feedback_included_in_prompt(self, basic_llm):
        tool = SectionWriterTool(llm=basic_llm)
        tool._run(section_name="introduction", feedback="Add more detail about methodology")

        prompt = basic_llm.invoke.call_args[0][0]
        assert "Revision Feedback" in prompt
        assert "Add more detail about methodology" in prompt

    def test_empty_feedback_not_in_prompt(self, basic_llm):
        tool = SectionWriterTool(llm=basic_llm)
        tool._run(section_name="introduction", feedback="")

        prompt = basic_llm.invoke.call_args[0][0]
        assert "Revision Feedback" not in prompt

    def test_whitespace_feedback_not_in_prompt(self, basic_llm):
        tool = SectionWriterTool(llm=basic_llm)
        tool._run(section_name="introduction", feedback="   ")

        prompt = basic_llm.invoke.call_args[0][0]
        assert "Revision Feedback" not in prompt


class TestResponseParsing:
    """Test various LLM response formats are handled."""

    def test_json_code_block_response(self):
        data = make_section_response(content="Parsed from code block")
        response_text = f"```json\n{json.dumps(data)}\n```"
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = response_text
        mock_llm.invoke.return_value = mock_response

        tool = SectionWriterTool(llm=mock_llm)
        result = tool._run(section_name="introduction")
        assert result.content == "Parsed from code block"

    def test_plain_code_block_response(self):
        data = make_section_response(content="Parsed from plain block")
        response_text = f"```\n{json.dumps(data)}\n```"
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = response_text
        mock_llm.invoke.return_value = mock_response

        tool = SectionWriterTool(llm=mock_llm)
        result = tool._run(section_name="introduction")
        assert result.content == "Parsed from plain block"

    def test_invalid_json_falls_back_to_raw_text(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This is not valid JSON, just plain text."
        mock_llm.invoke.return_value = mock_response

        tool = SectionWriterTool(llm=mock_llm)
        result = tool._run(section_name="introduction")
        assert result.content == "This is not valid JSON, just plain text."
        assert result.section_type == SectionType.INTRODUCTION
        assert result.citations == []

    def test_response_without_content_attr(self):
        data = make_section_response(content="From string response")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = json.dumps(data)

        tool = SectionWriterTool(llm=mock_llm)
        result = tool._run(section_name="introduction")
        assert result.content == "From string response"

    def test_missing_title_uses_default(self):
        data = {"content": "Some content", "citations": []}
        llm = make_llm_response(data)
        tool = SectionWriterTool(llm=llm)
        result = tool._run(section_name="literature_review")
        assert result.title == "Literature Review"

    def test_missing_citations_defaults_to_empty(self):
        data = {"title": "Intro", "content": "Some content"}
        llm = make_llm_response(data)
        tool = SectionWriterTool(llm=llm)
        result = tool._run(section_name="introduction")
        assert result.citations == []

    def test_non_list_citations_defaults_to_empty(self):
        data = {"title": "Intro", "content": "Content", "citations": "not_a_list"}
        llm = make_llm_response(data)
        tool = SectionWriterTool(llm=llm)
        result = tool._run(section_name="introduction")
        assert result.citations == []
