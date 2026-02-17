"""Paper Generator Agent that orchestrates the research paper writing workflow."""

import logging
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from src.core.state_manager import StateManager
from src.models.schemas import PaperState
from src.tools.arxiv_search import ArxivSearchTool
from src.tools.folder_reader import FolderReaderTool
from src.tools.outline_builder import OutlineBuilderTool
from src.tools.pdf_writer import PDFWriterTool
from src.tools.reference_manager import ReferenceManagerTool
from src.tools.section_writer import SectionWriterTool
from src.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an AI research paper writing assistant. You help researchers write "
    "structured, high-quality academic papers. You have access to tools for:\n"
    "- Building paper outlines\n"
    "- Writing individual sections\n"
    "- Reading reference materials from folders\n"
    "- Managing citations and bibliographies\n"
    "- Exporting papers as PDF\n"
    "- Searching the web for latest information (DuckDuckGo)\n"
    "- Searching ArXiv for academic papers and preprints\n\n"
    "Guide the user through the paper writing process step by step. "
    "When a tool fails, explain the error clearly and suggest how to fix it."
)


def create_paper_agent(
    paper_state: PaperState,
    vector_store: Any,
) -> CompiledStateGraph:
    """Create a LangChain agent with all paper-writing tools bound.

    Uses langchain's create_agent with a MemorySaver checkpointer for
    conversation history persistence across invocations.

    Args:
        paper_state: The current paper state for tools that need it.
        vector_store: ChromaDB vector store for reference material retrieval.

    Returns:
        Configured CompiledStateGraph agent with all tools, memory, and error handling.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    tools = _create_tools(paper_state, vector_store, llm)
    checkpointer = MemorySaver()

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    return agent


def _create_tools(
    paper_state: PaperState,
    vector_store: Any,
    llm: Any,
) -> list:
    """Create instances of all paper-writing tools.

    Args:
        paper_state: Current paper state shared across tools.
        vector_store: Vector store for reference retrieval.
        llm: Language model for tools that need LLM access.

    Returns:
        List of configured LangChain tool instances.
    """
    reference_manager = ReferenceManagerTool(
        citations=dict(paper_state.citations),
        citation_style=paper_state.citation_style,
    )

    outline_builder = OutlineBuilderTool(
        llm=llm,
        vector_store=vector_store,
    )

    section_writer = SectionWriterTool(
        llm=llm,
        vector_store=vector_store,
        paper_state=paper_state,
    )

    folder_reader = FolderReaderTool(
        vector_store=vector_store,
    )

    pdf_writer = PDFWriterTool(
        paper_state=paper_state,
        reference_manager=reference_manager,
    )

    web_search = WebSearchTool()

    arxiv_search = ArxivSearchTool()

    return [
        outline_builder,
        section_writer,
        folder_reader,
        reference_manager,
        pdf_writer,
        web_search,
        arxiv_search,
    ]
