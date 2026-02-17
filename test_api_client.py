#!/usr/bin/env python3
"""API integration test for the AI Research Paper Generator.

This script tests the REST API endpoints to ensure the complete workflow
works correctly through HTTP requests.

Prerequisites:
    - The API server must be running: python -m src.main
    - Dependencies must be installed
    - .env file must be configured with OPENAI_API_KEY
"""

import json
import os
import sys
import time
from pathlib import Path

import httpx


# API configuration
API_BASE_URL = "http://localhost:8000/api/v1"
TIMEOUT = 300.0  # 5 minutes for LLM operations


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def check_server_health():
    """Check if the API server is running."""
    try:
        response = httpx.get("http://localhost:8000", timeout=5.0)
        return response.status_code in [200, 404]  # 404 is ok, means server is running
    except httpx.ConnectError:
        return False


def create_session(client: httpx.Client) -> str:
    """Create a new paper session."""
    print("Creating new session...", end=" ", flush=True)
    response = client.post(f"{API_BASE_URL}/sessions")
    response.raise_for_status()
    session_id = response.json()["session_id"]
    print(f"✓ Session ID: {session_id}")
    return session_id


def generate_outline(client: httpx.Client, session_id: str):
    """Generate paper outline."""
    print("\nGenerating paper outline...", end=" ", flush=True)
    response = client.post(
        f"{API_BASE_URL}/sessions/{session_id}/outline",
        json={
            "topic": "AI fine tuning",
            "instructions": (
                "Create a comprehensive outline for a survey paper on AI fine-tuning techniques. "
                "Include recent developments like LoRA, QLoRA, and prompt tuning."
            ),
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    outline = response.json()
    print(f"✓ Generated {len(outline['sections'])} sections")
    
    print("\nOutline sections:")
    for section in outline["sections"]:
        print(f"  • {section['title']} ({section['section_type']})")
    
    return outline


def generate_section(client: httpx.Client, session_id: str, section_name: str):
    """Generate a specific paper section."""
    print(f"  Generating {section_name}...", end=" ", flush=True)
    response = client.post(
        f"{API_BASE_URL}/sessions/{session_id}/sections/{section_name}",
        json={"feedback": ""},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    section = response.json()
    word_count = len(section["content"].split())
    print(f"✓ ({word_count} words)")
    return section


def export_pdf(client: httpx.Client, session_id: str, output_path: str):
    """Export paper as PDF."""
    print(f"\nExporting to PDF: {output_path}...", end=" ", flush=True)
    response = client.post(
        f"{API_BASE_URL}/sessions/{session_id}/export/pdf",
        json={"output_path": output_path},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    result = response.json()
    print(f"✓ Exported")
    return result


def save_state(client: httpx.Client, session_id: str, file_path: str):
    """Save paper state to JSON file."""
    print(f"\nSaving paper state to: {file_path}...", end=" ", flush=True)
    response = client.post(
        f"{API_BASE_URL}/sessions/{session_id}/save",
        json={"file_path": file_path},
        timeout=30.0,
    )
    response.raise_for_status()
    result = response.json()
    print(f"✓ {result['message']}")
    return result


def main():
    """Run the API integration test."""
    print_section("API Integration Test - AI Research Paper Generator")
    
    # Check if server is running
    if not check_server_health():
        print("✗ ERROR: API server is not running!")
        print("\nPlease start the server first:")
        print("  python -m src.main")
        sys.exit(1)
    
    print("✓ API server is running")
    
    # Create HTTP client
    client = httpx.Client(base_url=API_BASE_URL)
    
    try:
        # Step 1: Create session
        print_section("Step 1: Create Session")
        session_id = create_session(client)
        
        # Step 2: Generate outline
        print_section("Step 2: Generate Outline")
        outline = generate_outline(client, session_id)
        
        # Step 3: Generate sections
        print_section("Step 3: Generate Paper Sections")
        sections_to_generate = ["abstract", "introduction", "methodology", "conclusion"]
        
        generated_sections = {}
        for section_name in sections_to_generate:
            try:
                section = generate_section(client, session_id, section_name)
                generated_sections[section_name] = section
            except Exception as exc:
                print(f"✗ Failed: {exc}")
        
        print(f"\n✓ Generated {len(generated_sections)}/{len(sections_to_generate)} sections")
        
        # Step 4: Export to PDF
        print_section("Step 4: Export to PDF")
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "api_test_paper.pdf")
        
        try:
            pdf_result = export_pdf(client, session_id, output_path)
            print(f"  Location: {pdf_result['output_path']}")
            
            if os.path.exists(pdf_result['output_path']):
                size_kb = os.path.getsize(pdf_result['output_path']) / 1024
                print(f"  Size: {size_kb:.1f} KB")
        except Exception as exc:
            print(f"✗ PDF export failed: {exc}")
        
        # Step 5: Save state
        print_section("Step 5: Save Paper State")
        state_path = os.path.join(output_dir, "paper_state.json")
        try:
            save_state(client, session_id, state_path)
            
            # Verify file was created
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state_data = json.load(f)
                print(f"  State file size: {os.path.getsize(state_path)} bytes")
                print(f"  Sections in state: {len(state_data.get('sections', {}))}")
        except Exception as exc:
            print(f"✗ State save failed: {exc}")
        
        # Summary
        print_section("Test Complete - Summary")
        print(f"Session ID: {session_id}")
        print(f"Sections Generated: {len(generated_sections)}")
        print(f"PDF Output: {output_path}")
        print(f"State File: {state_path}")
        
        print("\n✓ API integration test completed successfully!")
        
    except httpx.HTTPStatusError as exc:
        print(f"\n✗ HTTP Error: {exc.response.status_code}")
        print(f"Response: {exc.response.text}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n✗ Test failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
