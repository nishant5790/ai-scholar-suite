"""Reference Manager tool for managing citations and generating bibliographies."""

import json
import logging
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.models.schemas import CitationMetadata, CitationStyle

logger = logging.getLogger(__name__)


class ReferenceManagerInput(BaseModel):
    """Input schema for the ReferenceManagerTool."""

    action: str = Field(description="Action to perform: 'add', 'bibliography', or 'marker'")
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Citation metadata (required for 'add' action)"
    )
    style: Optional[str] = Field(
        default=None, description="Citation style: 'apa', 'ieee', or 'mla' (for 'bibliography' action)"
    )
    citation_id: Optional[str] = Field(
        default=None, description="Citation ID (required for 'marker' action)"
    )


class ReferenceManagerTool(BaseTool):
    """Tool that manages citations and generates formatted bibliographies."""

    name: str = "reference_manager"
    description: str = (
        "Manages citations and references for a research paper. "
        "Supports adding citations, generating bibliographies in APA/IEEE/MLA styles, "
        "and producing in-text citation markers."
    )
    args_schema: Type[BaseModel] = ReferenceManagerInput

    citations: dict[str, CitationMetadata] = Field(default_factory=dict)
    insertion_order: list[str] = Field(default_factory=list)
    citation_style: CitationStyle = Field(default=CitationStyle.APA)

    def _run(self, action: str, metadata: Optional[dict[str, Any]] = None,
             style: Optional[str] = None, citation_id: Optional[str] = None) -> str:
        """Route to the appropriate method based on the action parameter.

        Args:
            action: One of 'add', 'bibliography', or 'marker'.
            metadata: Citation metadata dict (for 'add' action).
            style: Citation style string (for 'bibliography' action).
            citation_id: Citation ID (for 'marker' action).

        Returns:
            Result string from the corresponding action.
        """
        if action == "add":
            if metadata is None:
                return json.dumps({"error": "metadata is required for 'add' action"})
            citation = CitationMetadata(**metadata)
            cid = self.add_citation(citation)
            return json.dumps({"citation_id": cid})

        elif action == "bibliography":
            bib_style = CitationStyle(style) if style else CitationStyle.APA
            bibliography = self.generate_bibliography(bib_style)
            return bibliography

        elif action == "marker":
            if citation_id is None:
                return json.dumps({"error": "citation_id is required for 'marker' action"})
            marker = self.get_inline_marker(citation_id)
            return marker

        else:
            return json.dumps({"error": f"Unknown action: {action}. Use 'add', 'bibliography', or 'marker'."})

    def _find_duplicate(self, metadata: CitationMetadata) -> Optional[str]:
        """Check if a citation with the same author+title+year already exists.

        Returns:
            The existing citation_id if a duplicate is found, None otherwise.
        """
        for cid, existing in self.citations.items():
            if (existing.author == metadata.author
                    and existing.title == metadata.title
                    and existing.year == metadata.year):
                return cid
        return None

    def add_citation(self, metadata: CitationMetadata) -> str:
        """Store a citation and return its citation ID.

        If a citation with the same author+title+year already exists,
        returns the existing citation_id without creating a duplicate.

        Args:
            metadata: The citation metadata to store.

        Returns:
            The citation ID (existing or newly assigned).
        """
        duplicate_id = self._find_duplicate(metadata)
        if duplicate_id is not None:
            return duplicate_id

        cid = metadata.citation_id
        self.citations[cid] = metadata
        self.insertion_order.append(cid)
        return cid

    def generate_bibliography(self, style: CitationStyle = CitationStyle.APA) -> str:
        """Generate a formatted bibliography string for all stored citations.

        Args:
            style: The citation style to use (APA, IEEE, or MLA). Defaults to APA.

        Returns:
            A formatted bibliography string with one entry per line.
        """
        if not self.citations:
            return ""

        entries = []
        for idx, cid in enumerate(self.insertion_order):
            if cid not in self.citations:
                continue
            meta = self.citations[cid]
            entry = self._format_entry(meta, style, idx + 1)
            entries.append(entry)

        return "\n".join(entries)

    def _format_entry(self, meta: CitationMetadata, style: CitationStyle, number: int) -> str:
        """Format a single bibliography entry according to the given style.

        Args:
            meta: Citation metadata.
            style: Citation formatting style.
            number: The 1-based insertion order number (used for IEEE).

        Returns:
            Formatted bibliography entry string.
        """
        if style == CitationStyle.APA:
            return f"{meta.author} ({meta.year}). {meta.title}. {meta.source}."
        elif style == CitationStyle.IEEE:
            return f"[{number}] {meta.author}, \"{meta.title},\" {meta.source}, {meta.year}."
        elif style == CitationStyle.MLA:
            author = meta.author.rstrip(".")
            return f"{author}. \"{meta.title}.\" {meta.source}, {meta.year}."
        else:
            return f"{meta.author} ({meta.year}). {meta.title}. {meta.source}."

    def get_inline_marker(self, citation_id: str) -> str:
        """Generate an in-text citation marker for the given citation ID.

        Args:
            citation_id: The ID of the citation to generate a marker for.

        Returns:
            Formatted in-text citation marker string.

        Raises:
            ValueError: If the citation_id is not found.
        """
        if citation_id not in self.citations:
            raise ValueError(f"Citation ID not found: {citation_id}")

        meta = self.citations[citation_id]

        if self.citation_style == CitationStyle.APA:
            return f"({meta.author}, {meta.year})"
        elif self.citation_style == CitationStyle.IEEE:
            number = self.insertion_order.index(citation_id) + 1
            return f"[{number}]"
        elif self.citation_style == CitationStyle.MLA:
            return f"({meta.author} {meta.year})"
        else:
            return f"({meta.author}, {meta.year})"
