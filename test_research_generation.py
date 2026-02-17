#!/usr/bin/env python3
"""Test script for research paper generation with web search and ArXiv tools.

This script demonstrates the complete workflow of generating a research paper
on "AI fine tuning" using the new web search and ArXiv retrieval capabilities.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_settings
from src.models.schemas import PaperState, SectionType
from src.tools.arxiv_search import ArxivSearchTool
from src.tools.outline_builder import OutlineBuilderTool
from src.tools.pdf_writer import PDFWriterTool
from src.tools.section_writer import SectionWriterTool
from src.tools.web_search import WebSearchTool
from langchain_openai import ChatOpenAI


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def main():
    """Run the complete research paper generation workflow."""
    print_section("AI Research Paper Generator - Test Run")
    
    # Load settings
    settings = get_settings()
    
    # Verify API key is set
    if not settings.openai_api_key:
        print("ERROR: OPENAI_API_KEY not found in environment!")
        print("Please ensure the .env file is properly configured.")
        sys.exit(1)
    
    print(f"✓ OpenAI API key loaded (length: {len(settings.openai_api_key)})")
    print(f"✓ Model: {settings.llm_model}")
    print(f"✓ Output directory: {settings.output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(settings.output_dir, exist_ok=True)
    
    # Initialize LLM
    llm = ChatOpenAI(model=settings.llm_model, temperature=0.3)
    print(f"✓ LLM initialized")
    
    # Initialize paper state
    paper_state = PaperState(
        title="A Comprehensive Survey on AI Fine-Tuning Techniques",
        author="AI Research Assistant",
        topic="AI fine tuning",
    )
    
    # Step 1: Search ArXiv for academic papers
    print_section("Step 1: Searching ArXiv for Academic Papers")
    arxiv_tool = ArxivSearchTool()
    try:
        arxiv_results = arxiv_tool._run(query="AI fine tuning", max_docs=3)
        print(f"✓ Found {arxiv_results.total_papers} ArXiv papers:")
        for i, paper in enumerate(arxiv_results.papers, 1):
            print(f"\n  {i}. {paper.title}")
            print(f"     Authors: {paper.authors}")
            print(f"     Published: {paper.published}")
            print(f"     ArXiv ID: {paper.arxiv_id}")
            print(f"     URL: {paper.pdf_url}")
    except Exception as exc:
        print(f"✗ ArXiv search failed: {exc}")
        print("Continuing without ArXiv results...")
    
    # Step 2: Search the web for latest information
    print_section("Step 2: Searching Web for Latest Information")
    web_tool = WebSearchTool()
    try:
        web_results = web_tool._run(query="AI fine tuning 2026", max_results=5)
        print(f"✓ Found {web_results.total_results} web results:")
        for i, result in enumerate(web_results.results, 1):
            print(f"\n  {i}. {result['title']}")
            print(f"     URL: {result['url']}")
            print(f"     Snippet: {result['snippet'][:150]}...")
    except Exception as exc:
        print(f"✗ Web search failed: {exc}")
        print("Continuing without web results...")
    
    # Step 3: Generate paper outline
    print_section("Step 3: Generating Paper Outline")
    outline_tool = OutlineBuilderTool(llm=llm, vector_store=None)
    try:
        outline = outline_tool._run(
            topic="AI fine tuning",
            instructions=(
                "Create a comprehensive outline for a survey paper on AI fine-tuning techniques. "
                "Include recent developments and practical applications. "
                "Cover techniques like LoRA, QLoRA, prompt tuning, and prefix tuning."
            ),
        )
        paper_state.outline = outline
        print(f"✓ Generated outline with {len(outline.sections)} sections:")
        for section in outline.sections:
            print(f"\n  • {section.title} ({section.section_type.value})")
            for point in section.key_points[:3]:  # Show first 3 points
                print(f"    - {point}")
            if len(section.key_points) > 3:
                print(f"    ... and {len(section.key_points) - 3} more points")
    except Exception as exc:
        print(f"✗ Outline generation failed: {exc}")
        sys.exit(1)
    
    # Step 4: Generate each section
    print_section("Step 4: Generating Paper Sections")
    section_tool = SectionWriterTool(llm=llm, vector_store=None, paper_state=paper_state)
    
    sections_to_generate = [
        SectionType.ABSTRACT,
        SectionType.INTRODUCTION,
        SectionType.LITERATURE_REVIEW,
        SectionType.METHODOLOGY,
        SectionType.RESULTS,
        SectionType.DISCUSSION,
        SectionType.CONCLUSION,
    ]
    
    for section_type in sections_to_generate:
        print(f"\n  Generating {section_type.value}...", end=" ", flush=True)
        try:
            section_content = section_tool._run(
                section_name=section_type.value,
                feedback="",
            )
            paper_state.sections[section_type.value] = section_content
            word_count = len(section_content.content.split())
            print(f"✓ ({word_count} words)")
            
            # Show a preview of the content
            preview = section_content.content[:200].replace('\n', ' ')
            print(f"     Preview: {preview}...")
        except Exception as exc:
            print(f"✗ Failed: {exc}")
    
    print(f"\n✓ Generated {len(paper_state.sections)} sections")
    
    # Step 5: Export to PDF
    print_section("Step 5: Exporting to PDF")
    output_path = os.path.join(settings.output_dir, "ai_fine_tuning_research.pdf")
    pdf_tool = PDFWriterTool(paper_state=paper_state)
    
    try:
        result = pdf_tool._run(output_path=output_path)
        if result.startswith("Error:"):
            print(f"✗ {result}")
        else:
            print(f"✓ PDF generated successfully!")
            print(f"  Location: {result}")
            
            # Check file exists and get size
            if os.path.exists(result):
                size_kb = os.path.getsize(result) / 1024
                print(f"  Size: {size_kb:.1f} KB")
    except Exception as exc:
        print(f"✗ PDF export failed: {exc}")
    
    # Summary
    print_section("Test Complete - Summary")
    print(f"Paper Title: {paper_state.title}")
    print(f"Topic: {paper_state.topic}")
    print(f"Sections Generated: {len(paper_state.sections)}")
    print(f"Total Word Count: {sum(len(s.content.split()) for s in paper_state.sections.values())}")
    print(f"Output File: {output_path}")
    
    print("\n✓ Research paper generation test completed successfully!")
    print("\nNext steps:")
    print("  1. Review the generated PDF")
    print("  2. Test the API endpoints using test_api_client.py")
    print("  3. Start the server with: python -m src.main")


if __name__ == "__main__":
    main()
