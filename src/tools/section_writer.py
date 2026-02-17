"""Section Writer tool for generating research paper section content."""

import json
import logging
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.models.schemas import PaperState, SectionContent, SectionType

logger = logging.getLogger(__name__)

VALID_SECTION_TYPES = [st.value for st in SectionType]

SECTION_ROLES = {
    SectionType.ABSTRACT: "A concise summary of the entire paper, including the research problem, methods, key findings, and conclusions.",
    SectionType.INTRODUCTION: "Introduces the research topic, states the problem, provides background context, and outlines the paper's objectives and structure.",
    SectionType.LITERATURE_REVIEW: "Surveys and synthesizes existing research relevant to the topic, identifying gaps the current paper addresses.",
    SectionType.METHODOLOGY: "Describes the research design, methods, data collection, and analysis procedures used in the study.",
    SectionType.RESULTS: "Presents the findings of the research objectively, using data, tables, and figures as appropriate.",
    SectionType.DISCUSSION: "Interprets the results, discusses implications, compares with existing literature, and addresses limitations.",
    SectionType.CONCLUSION: "Summarizes the key findings, restates the significance, and suggests directions for future research.",
}

SECTION_PROMPT_TEMPLATE = """You are an academic research assistant writing a section of a research paper.

Section Type: {section_type}
Section Role: {section_role}
{outline_block}
{previous_sections_block}
{references_block}
{feedback_block}
Write the content for the "{section_type}" section. Use an academic tone appropriate for a research paper.

You MUST respond with valid JSON in exactly this format (no extra text):
{{
  "title": "<section title>",
  "content": "<the full section text>",
  "citations": ["<citation_id_1>", "<citation_id_2>"]
}}

The "citations" array should list IDs of any references you cite. If no citations are used, return an empty array."""


class SectionWriterInput(BaseModel):
    """Input schema for the SectionWriterTool."""

    section_name: str = Field(description="The section type to generate (e.g. 'abstract', 'introduction')")
    feedback: str = Field(default="", description="Optional feedback for revising the section")


class SectionWriterTool(BaseTool):
    """Tool that generates content for individual research paper sections using an LLM."""

    name: str = "section_writer"
    description: str = (
        "Generates well-written academic content for a specific section of a research paper. "
        "Supports section types: abstract, introduction, literature_review, methodology, "
        "results, discussion, conclusion. Can revise sections based on feedback."
    )
    args_schema: Type[BaseModel] = SectionWriterInput

    llm: Any = Field(description="Language model for generating section content")
    vector_store: Any = Field(default=None, description="Optional ChromaDB collection for reference context")
    paper_state: PaperState = Field(default_factory=PaperState, description="Current paper state for context")

    def _run(self, section_name: str, feedback: str = "") -> SectionContent:
        """Generate content for the specified paper section.

        Args:
            section_name: The section type to generate (must be a valid SectionType value).
            feedback: Optional feedback for revising the section.

        Returns:
            SectionContent with the generated text and citations.

        Raises:
            ValueError: If section_name is not a recognized section type.
        """
        section_type = self._validate_section_type(section_name)
        references = self._get_reference_context(section_type.value)
        prompt = self._build_prompt(section_type, references, feedback.strip())

        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)

        return self._parse_response(section_type, response_text)

    def _validate_section_type(self, section_name: str) -> SectionType:
        """Validate that the section name is a recognized SectionType.

        Args:
            section_name: The section name to validate.

        Returns:
            The corresponding SectionType enum value.

        Raises:
            ValueError: If the section name is not recognized.
        """
        try:
            return SectionType(section_name.strip().lower())
        except ValueError:
            raise ValueError(
                f"Unrecognized section type: '{section_name}'. "
                f"Valid section types are: {', '.join(VALID_SECTION_TYPES)}"
            )

    def _get_reference_context(self, query: str) -> str:
        """Query the vector store for relevant reference context.

        Args:
            query: The search query (section type or topic).

        Returns:
            A string of relevant context, or empty string if unavailable.
        """
        if self.vector_store is None:
            return ""

        try:
            search_query = query
            if self.paper_state.topic:
                search_query = f"{self.paper_state.topic} {query}"

            results = self.vector_store.query(query_texts=[search_query], n_results=5)
            if results and results.get("documents") and results["documents"][0]:
                docs = results["documents"][0]
                return "\n\n".join(docs)
        except Exception as e:
            logger.warning("Failed to query vector store: %s", str(e))

        return ""

    def _build_prompt(self, section_type: SectionType, references: str, feedback: str) -> str:
        """Build the LLM prompt with all available context.

        Args:
            section_type: The type of section to generate.
            references: Reference context from vector store.
            feedback: Optional revision feedback.

        Returns:
            Formatted prompt string.
        """
        section_role = SECTION_ROLES.get(section_type, "A section of the research paper.")

        outline_block = ""
        if self.paper_state.outline:
            outline_data = self.paper_state.outline.model_dump()
            outline_block = f"\nPaper Outline:\n{json.dumps(outline_data, indent=2)}\n"

        previous_sections_block = ""
        if self.paper_state.sections:
            prev_parts = []
            for name, sec in self.paper_state.sections.items():
                prev_parts.append(f"--- {sec.title} ({name}) ---\n{sec.content}")
            previous_sections_block = (
                f"\nPreviously Generated Sections:\n" + "\n\n".join(prev_parts) + "\n"
            )

        references_block = ""
        if references:
            references_block = (
                f"\nReference Materials (use these to inform your writing and cite where appropriate):\n{references}\n"
            )

        feedback_block = ""
        if feedback:
            feedback_block = (
                f"\nRevision Feedback (incorporate this feedback into the section):\n{feedback}\n"
            )

        return SECTION_PROMPT_TEMPLATE.format(
            section_type=section_type.value,
            section_role=section_role,
            outline_block=outline_block,
            previous_sections_block=previous_sections_block,
            references_block=references_block,
            feedback_block=feedback_block,
        )

    def _parse_response(self, section_type: SectionType, response_text: str) -> SectionContent:
        """Parse the LLM response into a SectionContent object.

        Args:
            section_type: The expected section type.
            response_text: Raw LLM response text.

        Returns:
            Parsed SectionContent.
        """
        text = response_text.strip()

        # Extract JSON from markdown code blocks if present
        if "```json" in text:
            text = text.split("```json", 1)[1]
            text = text.split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1]
            text = text.split("```", 1)[0]

        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON: %s", str(e))
            # Fall back to using the raw text as content
            return SectionContent(
                section_type=section_type,
                title=section_type.value.replace("_", " ").title(),
                content=response_text.strip(),
                citations=[],
            )

        title = data.get("title", section_type.value.replace("_", " ").title())
        content = data.get("content", "")
        citations = data.get("citations", [])

        # Ensure citations is a list of strings
        if not isinstance(citations, list):
            citations = []
        citations = [str(c) for c in citations if c]

        return SectionContent(
            section_type=section_type,
            title=title,
            content=content,
            citations=citations,
        )
