# AI Research Paper Generator

AI-powered system for generating structured academic research papers. Built on LangChain agents with OpenAI, it combines web search (DuckDuckGo), academic paper retrieval (ArXiv), and intelligent section writing to produce complete, formatted research papers as PDFs.

## Features

- **Paper Outline Generation** -- structured outlines with all standard academic sections
- **Section Writing** -- AI-generated content for each section with academic tone
- **Web Search** -- DuckDuckGo integration for the latest information
- **ArXiv Retrieval** -- academic paper search via `langchain-community` ArxivRetriever
- **Reference Management** -- citation tracking with APA, IEEE, and MLA formatting
- **PDF Export** -- formatted multi-page PDF output via ReportLab
- **REST API** -- full FastAPI server with session-based workflows
- **Reference Ingestion** -- reads local PDF, TXT, DOCX, and Markdown files into ChromaDB
- **Docker Support** -- multi-stage Dockerfile and docker-compose for deployment

## Quick Start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Configure

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-key-here
LLM_MODEL=gpt-4o-mini
```

### 3. Generate a Paper (script)

```bash
python test_research_generation.py
```

This searches ArXiv and the web for material on "AI fine tuning", builds an outline, writes all seven sections, and exports a PDF to `./output/ai_fine_tuning_research.pdf`.

### 4. Generate a Paper (API)

Start the server:

```bash
python -m src.main
```

Then in another terminal:

```bash
python test_api_client.py
```

Or call endpoints directly -- see [docs/api.md](docs/api.md) for the full reference.

## Project Structure

```
src/
  agents/paper_agent.py       LangChain agent orchestrating all tools
  api/server.py               FastAPI REST endpoints
  core/session_manager.py     In-memory session store
  core/state_manager.py       JSON serialization of paper state
  models/schemas.py           Pydantic data models
  tools/
    arxiv_search.py           ArXiv paper retrieval
    web_search.py             DuckDuckGo web search
    outline_builder.py        Structured outline generation
    section_writer.py         Per-section content writing
    folder_reader.py          Local reference file ingestion
    reference_manager.py      Citation and bibliography management
    pdf_writer.py             PDF document export
  config.py                   Settings via pydantic-settings + dotenv
  main.py                     Application entry point
tests/                        Unit and property-based tests
docs/                         Detailed documentation
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/architecture.md](docs/architecture.md) | System architecture and data flow |
| [docs/tools.md](docs/tools.md) | Detailed reference for all seven tools |
| [docs/api.md](docs/api.md) | REST API endpoint reference |
| [docs/configuration.md](docs/configuration.md) | Environment variables and settings |
| [docs/development.md](docs/development.md) | Development setup, testing, Docker |

## Dependencies

Core runtime dependencies (see `pyproject.toml` for versions):

| Package | Purpose |
|---------|---------|
| langchain, langchain-openai | Agent framework and OpenAI integration |
| langchain-community | ArXiv retriever |
| duckduckgo-search | Web search |
| arxiv | ArXiv API client |
| fastapi, uvicorn | REST API server |
| reportlab | PDF generation |
| chromadb | Vector store for reference materials |
| pydantic, pydantic-settings | Data models and configuration |
| python-dotenv | `.env` file loading |

## License

This project is part of an AI research initiative.
