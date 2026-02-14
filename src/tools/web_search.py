"""Web Search Tool using DuckDuckGo for finding latest information."""

import logging
from typing import Any, Optional

from duckduckgo_search import DDGS
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    """Input schema for web search tool."""

    query: str = Field(..., description="The search query to look up")
    max_results: int = Field(
        default=5, description="Maximum number of search results to return", ge=1, le=10
    )


class WebSearchResult(BaseModel):
    """Result from web search."""

    query: str
    results: list[dict[str, str]]
    total_results: int


class WebSearchTool(BaseTool):
    """Tool for searching the web using DuckDuckGo.

    This tool retrieves the latest information from the web on any topic.
    It's particularly useful for finding recent developments, blog posts,
    articles, and other web content that might not be in academic databases.
    """

    name: str = "web_search"
    description: str = (
        "Search the web using DuckDuckGo for the latest information on a topic. "
        "Returns titles, URLs, and snippets from web search results. "
        "Use this for finding recent developments, blog posts, articles, and "
        "non-academic sources. For academic papers, use arxiv_search instead."
    )
    args_schema: type[BaseModel] = WebSearchInput

    def _run(
        self,
        query: str,
        max_results: int = 5,
        run_manager: Optional[Any] = None,
    ) -> WebSearchResult:
        """Execute web search and return results.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return (default: 5).
            run_manager: Optional callback manager (unused).

        Returns:
            WebSearchResult containing search results with titles, URLs, and snippets.

        Raises:
            ValueError: If query is empty or max_results is invalid.
            Exception: If search fails for any reason.
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        if max_results < 1 or max_results > 10:
            raise ValueError("max_results must be between 1 and 10")

        logger.info(f"Executing web search for: {query}")

        try:
            with DDGS() as ddgs:
                # Perform text search
                raw_results = list(ddgs.text(query, max_results=max_results))

            # Format results
            formatted_results = []
            for result in raw_results:
                formatted_results.append(
                    {
                        "title": result.get("title", "No title"),
                        "url": result.get("href", ""),
                        "snippet": result.get("body", "No description available"),
                    }
                )

            logger.info(f"Found {len(formatted_results)} web search results")

            return WebSearchResult(
                query=query,
                results=formatted_results,
                total_results=len(formatted_results),
            )

        except Exception as exc:
            logger.exception(f"Web search failed for query: {query}")
            raise Exception(f"Web search failed: {str(exc)}") from exc

    async def _arun(
        self,
        query: str,
        max_results: int = 5,
        run_manager: Optional[Any] = None,
    ) -> WebSearchResult:
        """Async version of web search (not implemented, falls back to sync)."""
        return self._run(query, max_results, run_manager)
