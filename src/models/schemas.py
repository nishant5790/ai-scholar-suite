"""Data models for the AI Research Paper Generator."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class SectionType(str, Enum):
    """Enum representing standard research paper section types."""

    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    LITERATURE_REVIEW = "literature_review"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"


class OutlineSection(BaseModel):
    """A single section within a paper outline."""

    section_type: SectionType
    title: str
    key_points: list[str]
    subsections: list["OutlineSection"] = []


class PaperOutline(BaseModel):
    """A structured outline for a research paper."""

    topic: str
    sections: list[OutlineSection]


class SectionContent(BaseModel):
    """Generated content for a paper section."""

    section_type: SectionType
    title: str
    content: str
    citations: list[str] = []  # list of citation IDs


class CitationMetadata(BaseModel):
    """Metadata for a single citation/reference."""

    citation_id: str
    author: str
    title: str
    year: int
    source: str
    doi: Optional[str] = None


class CitationStyle(str, Enum):
    """Supported citation formatting styles."""

    APA = "apa"
    IEEE = "ieee"
    MLA = "mla"


class PaperState(BaseModel):
    """Complete state of a research paper in progress."""

    title: str = ""
    author: str = ""
    topic: str = ""
    outline: Optional[PaperOutline] = None
    sections: dict[str, SectionContent] = {}
    citations: dict[str, CitationMetadata] = {}
    citation_style: CitationStyle = CitationStyle.APA


class IngestionResult(BaseModel):
    """Result of ingesting reference materials from a folder."""

    files_processed: int
    files_skipped: int
    skipped_files: list[str]
    total_chunks: int


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error: str
    message: str
    details: dict = {}
