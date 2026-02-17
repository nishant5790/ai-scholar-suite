"""Unit tests for the Paper Generator Agent."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.paper_agent import create_paper_agent, _create_tools
from src.models.schemas import (
    CitationMetadata,
    CitationStyle,
    PaperState,
)


@pytest.fixture
def paper_state():
    """A basic PaperState for testing."""
    return PaperState(
        title="Test Paper",
        author="Test Author",
        topic="Machine Learning",
    )


@pytest.fixture
def paper_state_with_citations():
    """PaperState with existing citations."""
    return PaperState(
        title="Test Paper",
        author="Test Author",
        topic="Machine Learning",
        citations={
            "ref1": CitationMetadata(
                citation_id="ref1",
                author="Smith, J.",
                title="A Study",
                year=2023,
                source="Journal of AI",
            )
        },
        citation_style=CitationStyle.IEEE,
    )


@pytest.fixture
def mock_vector_store():
    """A mock ChromaDB vector store."""
    return MagicMock()


class TestCreateTools:
    """Test that _create_tools produces the correct set of tool instances."""

    def test_returns_five_tools(self, paper_state, mock_vector_store):
        mock_llm = MagicMock()
        tools = _create_tools(paper_state, mock_vector_store, mock_llm)
        assert len(tools) == 5

    def test_tool_names(self, paper_state, mock_vector_store):
        mock_llm = MagicMock()
        tools = _create_tools(paper_state, mock_vector_store, mock_llm)
        names = {t.name for t in tools}
        expected = {"outline_builder", "section_writer", "folder_reader", "reference_manager", "pdf_writer"}
        assert names == expected

    def test_citations_passed_to_reference_manager(self, paper_state_with_citations, mock_vector_store):
        mock_llm = MagicMock()
        tools = _create_tools(paper_state_with_citations, mock_vector_store, mock_llm)
        ref_mgr = next(t for t in tools if t.name == "reference_manager")
        assert "ref1" in ref_mgr.citations

    def test_citation_style_passed_to_reference_manager(self, paper_state_with_citations, mock_vector_store):
        mock_llm = MagicMock()
        tools = _create_tools(paper_state_with_citations, mock_vector_store, mock_llm)
        ref_mgr = next(t for t in tools if t.name == "reference_manager")
        assert ref_mgr.citation_style == CitationStyle.IEEE

    def test_paper_state_passed_to_section_writer(self, paper_state, mock_vector_store):
        mock_llm = MagicMock()
        tools = _create_tools(paper_state, mock_vector_store, mock_llm)
        sw = next(t for t in tools if t.name == "section_writer")
        assert sw.paper_state.topic == "Machine Learning"

    def test_paper_state_passed_to_pdf_writer(self, paper_state, mock_vector_store):
        mock_llm = MagicMock()
        tools = _create_tools(paper_state, mock_vector_store, mock_llm)
        pw = next(t for t in tools if t.name == "pdf_writer")
        assert pw.paper_state.title == "Test Paper"

    def test_vector_store_passed_to_folder_reader(self, paper_state, mock_vector_store):
        mock_llm = MagicMock()
        tools = _create_tools(paper_state, mock_vector_store, mock_llm)
        fr = next(t for t in tools if t.name == "folder_reader")
        assert fr.vector_store is mock_vector_store

    def test_vector_store_passed_to_outline_builder(self, paper_state, mock_vector_store):
        mock_llm = MagicMock()
        tools = _create_tools(paper_state, mock_vector_store, mock_llm)
        ob = next(t for t in tools if t.name == "outline_builder")
        assert ob.vector_store is mock_vector_store

    def test_llm_passed_to_outline_builder(self, paper_state, mock_vector_store):
        mock_llm = MagicMock()
        tools = _create_tools(paper_state, mock_vector_store, mock_llm)
        ob = next(t for t in tools if t.name == "outline_builder")
        assert ob.llm is mock_llm

    def test_llm_passed_to_section_writer(self, paper_state, mock_vector_store):
        mock_llm = MagicMock()
        tools = _create_tools(paper_state, mock_vector_store, mock_llm)
        sw = next(t for t in tools if t.name == "section_writer")
        assert sw.llm is mock_llm


class TestCreatePaperAgent:
    """Test the full create_paper_agent function with mocked LLM."""

    @patch("src.agents.paper_agent.create_agent")
    @patch("src.agents.paper_agent.ChatOpenAI")
    def test_returns_compiled_graph(self, mock_chat, mock_create_agent, paper_state, mock_vector_store):
        mock_chat.return_value = MagicMock()
        mock_graph = MagicMock()
        mock_create_agent.return_value = mock_graph

        result = create_paper_agent(paper_state, mock_vector_store)

        assert result is mock_graph

    @patch("src.agents.paper_agent.create_agent")
    @patch("src.agents.paper_agent.ChatOpenAI")
    def test_agent_created_with_all_tools(self, mock_chat, mock_create_agent, paper_state, mock_vector_store):
        mock_chat.return_value = MagicMock()
        mock_create_agent.return_value = MagicMock()

        create_paper_agent(paper_state, mock_vector_store)

        call_kwargs = mock_create_agent.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools")
        tool_names = {t.name for t in tools}
        expected = {"outline_builder", "section_writer", "folder_reader", "reference_manager", "pdf_writer"}
        assert tool_names == expected

    @patch("src.agents.paper_agent.create_agent")
    @patch("src.agents.paper_agent.ChatOpenAI")
    def test_memory_checkpointer_configured(self, mock_chat, mock_create_agent, paper_state, mock_vector_store):
        mock_chat.return_value = MagicMock()
        mock_create_agent.return_value = MagicMock()

        create_paper_agent(paper_state, mock_vector_store)

        call_kwargs = mock_create_agent.call_args
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = call_kwargs.kwargs.get("checkpointer") or call_kwargs[1].get("checkpointer")
        assert isinstance(checkpointer, MemorySaver)

    @patch("src.agents.paper_agent.create_agent")
    @patch("src.agents.paper_agent.ChatOpenAI")
    def test_system_prompt_configured(self, mock_chat, mock_create_agent, paper_state, mock_vector_store):
        mock_chat.return_value = MagicMock()
        mock_create_agent.return_value = MagicMock()

        create_paper_agent(paper_state, mock_vector_store)

        call_kwargs = mock_create_agent.call_args
        system_prompt = call_kwargs.kwargs.get("system_prompt") or call_kwargs[1].get("system_prompt")
        assert system_prompt is not None
        assert "research paper" in system_prompt.lower()

    @patch("src.agents.paper_agent.create_agent")
    @patch("src.agents.paper_agent.ChatOpenAI")
    def test_llm_created_with_expected_params(self, mock_chat, mock_create_agent, paper_state, mock_vector_store):
        mock_chat.return_value = MagicMock()
        mock_create_agent.return_value = MagicMock()

        create_paper_agent(paper_state, mock_vector_store)

        mock_chat.assert_called_once_with(model="gpt-4o-mini", temperature=0.3)

    @patch("src.agents.paper_agent.create_agent")
    @patch("src.agents.paper_agent.ChatOpenAI")
    def test_llm_passed_as_model(self, mock_chat, mock_create_agent, paper_state, mock_vector_store):
        mock_llm_instance = MagicMock()
        mock_chat.return_value = mock_llm_instance
        mock_create_agent.return_value = MagicMock()

        create_paper_agent(paper_state, mock_vector_store)

        call_kwargs = mock_create_agent.call_args
        model = call_kwargs.kwargs.get("model") or call_kwargs[1].get("model")
        assert model is mock_llm_instance
