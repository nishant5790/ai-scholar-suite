"""ArXiv Search Tool for retrieving academic papers."""

import logging
from typing import Any, Optional

from langchain.tools import BaseTool
from langchain_community.retrievers import ArxivRetriever
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ArxivSearchInput(BaseModel):
    """Input schema for ArXiv search tool."""

    query: str = Field(..., description="The search query for ArXiv papers")
    max_docs: int = Field(
        default=2,
        description="Maximum number of papers to retrieve (default: 2)",
        ge=1,
        le=10,
    )


class ArxivPaper(BaseModel):
    """Metadata for a single ArXiv paper."""

    title: str
    authors: str
    published: str
    arxiv_id: str
    summary: str
    pdf_url: str
    entry_id: str


class ArxivSearchResult(BaseModel):
    """Result from ArXiv search."""

    query: str
    papers: list[ArxivPaper]
    total_papers: int


class ArxivSearchTool(BaseTool):
    """Tool for searching and retrieving academic papers from ArXiv.

    This tool searches ArXiv for academic papers on a given topic and retrieves
    full paper metadata including titles, authors, abstracts, and URLs. It's ideal
    for finding authoritative academic sources for research papers.
    """

    name: str = "arxiv_search"
    description: str = (
        "Search ArXiv for academic papers and retrieve full documents. "
        "Returns paper metadata including titles, authors, abstracts, publication dates, "
        "and ArXiv IDs. Use this for finding peer-reviewed academic papers and preprints. "
        "For general web content, use web_search instead."
    )
    args_schema: type[BaseModel] = ArxivSearchInput

    def _run(
        self,
        query: str,
        max_docs: int = 2,
        run_manager: Optional[Any] = None,
    ) -> ArxivSearchResult:
        """Execute ArXiv search and return paper results.

        Args:
            query: The search query string.
            max_docs: Maximum number of papers to retrieve (default: 2).
            run_manager: Optional callback manager (unused).

        Returns:
            ArxivSearchResult containing paper metadata.

        Raises:
            ValueError: If query is empty or max_docs is invalid.
            Exception: If search fails for any reason.
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        if max_docs < 1 or max_docs > 10:
            raise ValueError("max_docs must be between 1 and 10")

        logger.info(f"Executing ArXiv search for: {query}")

        try:
            # Create ArXiv retriever
            # Set get_full_documents=False to avoid requiring PyMuPDF
            # If you want full document text, install: pip install pymupdf
            retriever = ArxivRetriever(
                load_max_docs=max_docs,
                get_full_documents=False,  # Set to True if pymupdf is installed
            )

            # Retrieve documents
            docs = retriever.invoke(query)

            # Format results
            papers = []
            for doc in docs:
                metadata = doc.metadata
                
                # Convert published date to string if it's a date object
                published = metadata.get("Published", "Unknown date")
                if hasattr(published, 'strftime'):
                    published = published.strftime("%Y-%m-%d")
                elif not isinstance(published, str):
                    published = str(published)
                
                papers.append(
                    ArxivPaper(
                        title=metadata.get("Title", "No title"),
                        authors=metadata.get("Authors", "Unknown"),
                        published=published,
                        arxiv_id=metadata.get("Entry ID", "").split("/")[-1],
                        summary=metadata.get("Summary", "No summary available"),
                        pdf_url=metadata.get("pdf_url", ""),
                        entry_id=metadata.get("Entry ID", ""),
                    )
                )

            logger.info(f"Found {len(papers)} ArXiv papers")

            return ArxivSearchResult(
                query=query,
                papers=papers,
                total_papers=len(papers),
            )

        except Exception as exc:
            logger.exception(f"ArXiv search failed for query: {query}")
            raise Exception(f"ArXiv search failed: {str(exc)}") from exc

    async def _arun(
        self,
        query: str,
        max_docs: int = 2,
        run_manager: Optional[Any] = None,
    ) -> ArxivSearchResult:
        """Async version of ArXiv search (not implemented, falls back to sync)."""
        return self._run(query, max_docs, run_manager)
