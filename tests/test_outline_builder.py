"""Unit tests for the OutlineBuilderTool."""

import json
from unittest.mock import MagicMock

import pytest

from src.models.schemas import OutlineSection, PaperOutline, SectionType
from src.tools.outline_builder import REQUIRED_SECTIONS, OutlineBuilderTool


def make_llm_response(sections_data: list[dict]) -> MagicMock:
    """Create a mock LLM that returns a JSON response with the given sections."""
    response_json = json.dumps({"sections": sections_data})
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response_json
    mock_llm.invoke.return_value = mock_response
    return mock_llm


def make_complete_sections_data() -> list[dict]:
    """Create section data for all 7 required sections."""
    return [
        {
            "section_type": st.value,
            "title": st.value.replace("_", " ").title(),
            "key_points": [f"Key point for {st.value}"],
            "subsections": [],
        }
        for st in REQUIRED_SECTIONS
    ]


@pytest.fixture
def complete_llm():
    """LLM mock that returns a complete outline with all sections."""
    return make_llm_response(make_complete_sections_data())


@pytest.fixture
def tool_with_llm(complete_llm):
    """OutlineBuilderTool with a mock LLM returning complete sections."""
    return OutlineBuilderTool(llm=complete_llm)


class TestEmptyTopicValidation:
    """Requirement 1.4: Return error for empty/missing topic."""

    def test_empty_string_raises(self, tool_with_llm):
        with pytest.raises(ValueError, match="topic is required"):
            tool_with_llm._run(topic="")

    def test_whitespace_only_raises(self, tool_with_llm):
        with pytest.raises(ValueError, match="topic is required"):
            tool_with_llm._run(topic="   ")

    def test_valid_topic_does_not_raise(self, tool_with_llm):
        result = tool_with_llm._run(topic="Machine Learning")
        assert isinstance(result, PaperOutline)


class TestOutlineStructure:
    """Requirement 1.1: Outline contains all standard sections."""

    def test_contains_all_required_sections(self, tool_with_llm):
        result = tool_with_llm._run(topic="Deep Learning")
        section_types = {s.section_type for s in result.sections}
        for req in REQUIRED_SECTIONS:
            assert req in section_types

    def test_sections_have_nonempty_titles(self, tool_with_llm):
        result = tool_with_llm._run(topic="Deep Learning")
        for section in result.sections:
            assert section.title.strip() != ""

    def test_sections_have_key_points(self, tool_with_llm):
        result = tool_with_llm._run(topic="Deep Learning")
        for section in result.sections:
            assert len(section.key_points) >= 1

    def test_topic_preserved_in_outline(self, tool_with_llm):
        result = tool_with_llm._run(topic="Quantum Computing")
        assert result.topic == "Quantum Computing"

    def test_sections_in_standard_order(self, tool_with_llm):
        result = tool_with_llm._run(topic="Deep Learning")
        types = [s.section_type for s in result.sections]
        expected_order = list(REQUIRED_SECTIONS)
        # Filter to only required types for comparison
        filtered = [t for t in types if t in expected_order]
        assert filtered == expected_order


class TestMissingSectionsFilled:
    """Outline builder fills in missing sections from LLM response."""

    def test_missing_sections_are_added(self):
        # LLM only returns 3 sections
        partial_data = [
            {"section_type": "abstract", "title": "Abstract", "key_points": ["Overview"]},
            {"section_type": "introduction", "title": "Introduction", "key_points": ["Background"]},
            {"section_type": "conclusion", "title": "Conclusion", "key_points": ["Summary"]},
        ]
        llm = make_llm_response(partial_data)
        tool = OutlineBuilderTool(llm=llm)
        result = tool._run(topic="Test Topic")

        section_types = {s.section_type for s in result.sections}
        for req in REQUIRED_SECTIONS:
            assert req in section_types

    def test_invalid_json_falls_back_to_default(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This is not valid JSON at all"
        mock_llm.invoke.return_value = mock_response
        tool = OutlineBuilderTool(llm=mock_llm)

        result = tool._run(topic="Fallback Test")
        section_types = {s.section_type for s in result.sections}
        for req in REQUIRED_SECTIONS:
            assert req in section_types


class TestInstructionsPassthrough:
    """Requirement 1.3: Instructions are included in the LLM prompt."""

    def test_instructions_included_in_prompt(self, complete_llm):
        tool = OutlineBuilderTool(llm=complete_llm)
        tool._run(topic="AI Ethics", instructions="Focus on bias in algorithms")

        call_args = complete_llm.invoke.call_args[0][0]
        assert "Focus on bias in algorithms" in call_args

    def test_empty_instructions_not_in_prompt(self, complete_llm):
        tool = OutlineBuilderTool(llm=complete_llm)
        tool._run(topic="AI Ethics", instructions="")

        call_args = complete_llm.invoke.call_args[0][0]
        assert "Additional Instructions" not in call_args


class TestVectorStoreIntegration:
    """Requirement 1.2: Reference context from vector store is used."""

    def test_vector_store_queried_when_available(self, complete_llm):
        mock_vs = MagicMock()
        mock_vs.query.return_value = {
            "documents": [["Relevant context about neural networks"]],
        }
        tool = OutlineBuilderTool(llm=complete_llm, vector_store=mock_vs)
        tool._run(topic="Neural Networks")

        mock_vs.query.assert_called_once()
        call_args = complete_llm.invoke.call_args[0][0]
        assert "Relevant context about neural networks" in call_args

    def test_no_vector_store_still_works(self, complete_llm):
        tool = OutlineBuilderTool(llm=complete_llm, vector_store=None)
        result = tool._run(topic="Neural Networks")
        assert isinstance(result, PaperOutline)

    def test_vector_store_error_handled_gracefully(self, complete_llm):
        mock_vs = MagicMock()
        mock_vs.query.side_effect = RuntimeError("Connection failed")
        tool = OutlineBuilderTool(llm=complete_llm, vector_store=mock_vs)

        result = tool._run(topic="Neural Networks")
        assert isinstance(result, PaperOutline)


class TestJsonParsing:
    """Test various LLM response formats are handled."""

    def test_markdown_code_block_json(self):
        sections = make_complete_sections_data()
        response_text = f"```json\n{json.dumps({'sections': sections})}\n```"
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = response_text
        mock_llm.invoke.return_value = mock_response

        tool = OutlineBuilderTool(llm=mock_llm)
        result = tool._run(topic="Test")
        assert len(result.sections) == 7

    def test_plain_code_block(self):
        sections = make_complete_sections_data()
        response_text = f"```\n{json.dumps({'sections': sections})}\n```"
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = response_text
        mock_llm.invoke.return_value = mock_response

        tool = OutlineBuilderTool(llm=mock_llm)
        result = tool._run(topic="Test")
        assert len(result.sections) == 7

    def test_response_without_content_attr(self):
        """Handle LLM responses that return plain strings."""
        sections = make_complete_sections_data()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = json.dumps({"sections": sections})

        tool = OutlineBuilderTool(llm=mock_llm)
        result = tool._run(topic="Test")
        assert isinstance(result, PaperOutline)
