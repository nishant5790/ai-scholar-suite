"""PDF Writer tool for generating formatted research paper PDFs."""

import logging
from datetime import datetime
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
)

from src.models.schemas import PaperState, SectionType
from src.tools.reference_manager import ReferenceManagerTool

logger = logging.getLogger(__name__)

REQUIRED_SECTIONS = [
    SectionType.ABSTRACT,
    SectionType.INTRODUCTION,
    SectionType.LITERATURE_REVIEW,
    SectionType.METHODOLOGY,
    SectionType.RESULTS,
    SectionType.DISCUSSION,
    SectionType.CONCLUSION,
]

# Section rendering order
SECTION_ORDER = [
    SectionType.ABSTRACT,
    SectionType.INTRODUCTION,
    SectionType.LITERATURE_REVIEW,
    SectionType.METHODOLOGY,
    SectionType.RESULTS,
    SectionType.DISCUSSION,
    SectionType.CONCLUSION,
]


class PDFWriterInput(BaseModel):
    """Input schema for the PDFWriterTool."""

    output_path: str = Field(description="File path where the PDF should be saved")


class PDFWriterTool(BaseTool):
    """Tool that compiles paper content into a formatted PDF document."""

    name: str = "pdf_writer"
    description: str = (
        "Generates a formatted PDF document from the current paper state. "
        "Includes title page, all sections with consistent formatting, "
        "and bibliography when citations exist."
    )
    args_schema: Type[BaseModel] = PDFWriterInput

    paper_state: PaperState = Field(default_factory=PaperState)
    reference_manager: Optional[ReferenceManagerTool] = Field(default=None)

    def _run(self, output_path: str) -> str:
        """Generate a formatted PDF from the current paper state.

        Args:
            output_path: File path where the PDF should be written.

        Returns:
            The output file path on success, or an error message listing
            missing sections when the paper is incomplete.
        """
        missing = self._get_missing_sections()
        if missing:
            missing_names = [s.value for s in missing]
            return f"Error: Missing required sections: {', '.join(missing_names)}"

        try:
            self._build_pdf(output_path)
            return output_path
        except Exception as e:
            logger.error("Failed to generate PDF: %s", e)
            return f"Error: Failed to generate PDF: {e}"

    def _get_missing_sections(self) -> list[SectionType]:
        """Return a list of required sections not present in paper_state."""
        missing = []
        for section_type in REQUIRED_SECTIONS:
            if section_type.value not in self.paper_state.sections:
                missing.append(section_type)
        return missing

    def _build_pdf(self, output_path: str) -> None:
        """Build the PDF document and write it to the given path."""
        styles = self._create_styles()
        doc = self._create_document(output_path)
        story = []

        # Title page
        story.extend(self._build_title_page(styles))
        story.append(NextPageTemplate("content"))
        story.append(PageBreak())

        # Sections
        for section_type in SECTION_ORDER:
            section = self.paper_state.sections.get(section_type.value)
            if section:
                story.extend(self._build_section(section.title, section.content, styles))

        # Bibliography
        bib_text = self._generate_bibliography()
        if bib_text:
            story.extend(self._build_bibliography(bib_text, styles))

        doc.build(story)

    def _create_styles(self) -> dict[str, ParagraphStyle]:
        """Create and return the paragraph styles used in the PDF."""
        base = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "PaperTitle",
            parent=base["Title"],
            fontSize=24,
            leading=30,
            spaceAfter=20,
            alignment=1,  # center
        )

        author_style = ParagraphStyle(
            "PaperAuthor",
            parent=base["Normal"],
            fontSize=14,
            leading=18,
            spaceAfter=10,
            alignment=1,
        )

        date_style = ParagraphStyle(
            "PaperDate",
            parent=base["Normal"],
            fontSize=12,
            leading=16,
            spaceAfter=10,
            alignment=1,
        )

        heading_style = ParagraphStyle(
            "SectionHeading",
            parent=base["Heading1"],
            fontSize=16,
            leading=20,
            spaceBefore=20,
            spaceAfter=10,
            fontName="Helvetica-Bold",
        )

        body_style = ParagraphStyle(
            "SectionBody",
            parent=base["Normal"],
            fontSize=11,
            leading=15,
            spaceAfter=8,
            fontName="Helvetica",
        )

        bib_heading_style = ParagraphStyle(
            "BibHeading",
            parent=base["Heading1"],
            fontSize=16,
            leading=20,
            spaceBefore=20,
            spaceAfter=10,
            fontName="Helvetica-Bold",
        )

        bib_entry_style = ParagraphStyle(
            "BibEntry",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            spaceAfter=6,
            fontName="Helvetica",
            leftIndent=36,
            firstLineIndent=-36,
        )

        return {
            "title": title_style,
            "author": author_style,
            "date": date_style,
            "heading": heading_style,
            "body": body_style,
            "bib_heading": bib_heading_style,
            "bib_entry": bib_entry_style,
        }

    def _create_document(self, output_path: str) -> BaseDocTemplate:
        """Create the PDF document template with page numbering."""
        page_width, page_height = letter
        margin = inch

        title_frame = Frame(
            margin,
            margin,
            page_width - 2 * margin,
            page_height - 2 * margin,
            id="title_frame",
        )

        content_frame = Frame(
            margin,
            margin + 0.5 * inch,  # extra bottom margin for page number
            page_width - 2 * margin,
            page_height - 2 * margin - 0.5 * inch,
            id="content_frame",
        )

        def _add_page_number(canvas, doc):
            canvas.saveState()
            canvas.setFont("Helvetica", 9)
            page_num = canvas.getPageNumber()
            text = f"{page_num}"
            canvas.drawCentredString(page_width / 2, 0.5 * inch, text)
            canvas.restoreState()

        title_template = PageTemplate(
            id="title",
            frames=[title_frame],
        )

        content_template = PageTemplate(
            id="content",
            frames=[content_frame],
            onPage=_add_page_number,
        )

        doc = BaseDocTemplate(
            output_path,
            pagesize=letter,
            leftMargin=margin,
            rightMargin=margin,
            topMargin=margin,
            bottomMargin=margin,
        )
        doc.addPageTemplates([title_template, content_template])
        return doc

    def _build_title_page(self, styles: dict[str, ParagraphStyle]) -> list:
        """Build the title page elements."""
        elements = []
        elements.append(Spacer(1, 2 * inch))

        title = self.paper_state.title or "Untitled Paper"
        elements.append(Paragraph(title, styles["title"]))
        elements.append(Spacer(1, 0.5 * inch))

        author = self.paper_state.author or "Unknown Author"
        elements.append(Paragraph(author, styles["author"]))
        elements.append(Spacer(1, 0.25 * inch))

        date_str = datetime.now().strftime("%B %d, %Y")
        elements.append(Paragraph(date_str, styles["date"]))

        return elements

    def _build_section(self, title: str, content: str, styles: dict[str, ParagraphStyle]) -> list:
        """Build elements for a single paper section."""
        elements = []
        elements.append(Paragraph(title, styles["heading"]))

        for paragraph in content.split("\n\n"):
            paragraph = paragraph.strip()
            if paragraph:
                elements.append(Paragraph(paragraph, styles["body"]))

        return elements

    def _build_bibliography(self, bib_text: str, styles: dict[str, ParagraphStyle]) -> list:
        """Build bibliography elements."""
        elements = []
        elements.append(Paragraph("References", styles["bib_heading"]))

        for entry in bib_text.split("\n"):
            entry = entry.strip()
            if entry:
                elements.append(Paragraph(entry, styles["bib_entry"]))

        return elements

    def _generate_bibliography(self) -> str:
        """Generate bibliography text from the reference manager or paper state citations."""
        if self.reference_manager and self.reference_manager.citations:
            return self.reference_manager.generate_bibliography(
                self.paper_state.citation_style
            )

        if self.paper_state.citations:
            ref_mgr = ReferenceManagerTool()
            for cid, meta in self.paper_state.citations.items():
                ref_mgr.add_citation(meta)
            return ref_mgr.generate_bibliography(self.paper_state.citation_style)

        return ""
