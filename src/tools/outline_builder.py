"""Outline Builder tool for generating structured research paper outlines."""

import json
import logging
from typing import Any, Optional, Type

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.models.schemas import OutlineSection, PaperOutline, SectionType

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

OUTLINE_PROMPT_TEMPLATE = """You are an academic research assistant. Generate a structured outline for a research paper on the following topic.

Topic: {topic}
{instructions_block}
{context_block}
You MUST respond with valid JSON in exactly this format (no extra text):
{{
  "sections": [
    {{
      "section_type": "<one of: abstract, introduction, literature_review, methodology, results, discussion, conclusion>",
      "title": "<section title>",
      "key_points": ["<point 1>", "<point 2>"],
      "subsections": []
    }}
  ]
}}

You MUST include ALL of these section types: abstract, introduction, literature_review, methodology, results, discussion, conclusion.
Each section must have a non-empty title and at least one key point."""


class OutlineBuilderInput(BaseModel):
    """Input schema for the OutlineBuilderTool."""

    topic: str = Field(description="The research topic for the paper outline")
    instructions: str = Field(default="", description="Optional additional instructions or constraints")


class OutlineBuilderTool(BaseTool):
    """Tool that generates structured research paper outlines using an LLM."""

    name: str = "outline_builder"
    description: str = (
        "Generates a structured outline for a research paper based on a topic "
        "and optional instructions. The outline includes standard academic sections: "
        "abstract, introduction, literature review, methodology, results, discussion, conclusion."
    )
    args_schema: Type[BaseModel] = OutlineBuilderInput

    llm: Any = Field(description="Language model for generating outlines")
    vector_store: Any = Field(default=None, description="Optional vector store for reference context")

    def _run(self, topic: str, instructions: str = "") -> PaperOutline:
        """Generate a structured paper outline for the given topic.

        Args:
            topic: The research topic for the paper.
            instructions: Optional additional instructions or constraints.

        Returns:
            PaperOutline with all standard sections.

        Raises:
            ValueError: If the topic is empty or missing.
        """
        if not topic or not topic.strip():
            raise ValueError("A research topic is required to generate an outline.")

        context = self._get_reference_context(topic)
        prompt = self._build_prompt(topic.strip(), instructions.strip(), context)

        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)

        outline = self._parse_response(topic.strip(), response_text)
        self._validate_outline(outline)

        return outline

    def _get_reference_context(self, topic: str) -> str:
        """Query the vector store for relevant reference context.

        Args:
            topic: The research topic to search for.

        Returns:
            A string of relevant context, or empty string if no vector store.
        """
        if self.vector_store is None:
            return ""

        try:
            results = self.vector_store.query(query_texts=[topic], n_results=5)
            if results and results.get("documents") and results["documents"][0]:
                docs = results["documents"][0]
                return "\n\n".join(docs)
        except Exception as e:
            logger.warning("Failed to query vector store: %s", str(e))

        return ""

    def _build_prompt(self, topic: str, instructions: str, context: str) -> str:
        """Build the LLM prompt with topic, instructions, and context.

        Args:
            topic: The research topic.
            instructions: Additional instructions.
            context: Reference context from vector store.

        Returns:
            Formatted prompt string.
        """
        instructions_block = ""
        if instructions:
            instructions_block = f"\nAdditional Instructions: {instructions}\n"

        context_block = ""
        if context:
            context_block = (
                f"\nReference Context (use these to inform the outline):\n{context}\n"
            )

        return OUTLINE_PROMPT_TEMPLATE.format(
            topic=topic,
            instructions_block=instructions_block,
            context_block=context_block,
        )

    def _parse_response(self, topic: str, response_text: str) -> PaperOutline:
        """Parse the LLM response into a PaperOutline.

        Args:
            topic: The original research topic.
            response_text: Raw LLM response text.

        Returns:
            Parsed PaperOutline.
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
            return self._build_default_outline(topic)

        sections = []
        raw_sections = data.get("sections", [])

        for raw in raw_sections:
            try:
                section_type = SectionType(raw["section_type"])
                title = raw.get("title", section_type.value.replace("_", " ").title())
                key_points = raw.get("key_points", ["Key point to be developed"])
                subsections_raw = raw.get("subsections", [])

                subsections = []
                for sub in subsections_raw:
                    try:
                        sub_section = OutlineSection(
                            section_type=SectionType(sub["section_type"]),
                            title=sub.get("title", ""),
                            key_points=sub.get("key_points", []),
                            subsections=[],
                        )
                        subsections.append(sub_section)
                    except (KeyError, ValueError):
                        continue

                section = OutlineSection(
                    section_type=section_type,
                    title=title,
                    key_points=key_points if key_points else ["Key point to be developed"],
                    subsections=subsections,
                )
                sections.append(section)
            except (KeyError, ValueError) as e:
                logger.warning("Skipping invalid section: %s", str(e))
                continue

        # Fill in any missing required sections
        present_types = {s.section_type for s in sections}
        for req_type in REQUIRED_SECTIONS:
            if req_type not in present_types:
                sections.append(
                    OutlineSection(
                        section_type=req_type,
                        title=req_type.value.replace("_", " ").title(),
                        key_points=["Key point to be developed"],
                        subsections=[],
                    )
                )

        # Sort sections in standard order
        order = {st: i for i, st in enumerate(REQUIRED_SECTIONS)}
        sections.sort(key=lambda s: order.get(s.section_type, len(REQUIRED_SECTIONS)))

        return PaperOutline(topic=topic, sections=sections)

    def _build_default_outline(self, topic: str) -> PaperOutline:
        """Build a default outline with all required sections when parsing fails.

        Args:
            topic: The research topic.

        Returns:
            PaperOutline with default sections.
        """
        sections = [
            OutlineSection(
                section_type=st,
                title=st.value.replace("_", " ").title(),
                key_points=["Key point to be developed"],
                subsections=[],
            )
            for st in REQUIRED_SECTIONS
        ]
        return PaperOutline(topic=topic, sections=sections)

    def _validate_outline(self, outline: PaperOutline) -> None:
        """Validate that the outline contains all required sections.

        Args:
            outline: The outline to validate.

        Raises:
            ValueError: If required sections are missing.
        """
        present_types = {s.section_type for s in outline.sections}
        missing = [st.value for st in REQUIRED_SECTIONS if st not in present_types]

        if missing:
            raise ValueError(
                f"Outline is missing required sections: {', '.join(missing)}"
            )

        for section in outline.sections:
            if not section.title or not section.title.strip():
                raise ValueError(
                    f"Section '{section.section_type.value}' has an empty title."
                )
            if not section.key_points:
                raise ValueError(
                    f"Section '{section.section_type.value}' has no key points."
                )
