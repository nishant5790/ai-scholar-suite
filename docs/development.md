# Development Guide

## Prerequisites

- Python 3.11 or higher
- An OpenAI API key
- Internet access (for web search and ArXiv retrieval)

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install in editable mode with dev dependencies

```bash
pip install -e ".[dev]"
```

This installs the project plus testing tools (pytest, hypothesis, httpx, pytest-asyncio).

### 3. Configure environment

Copy the example and fill in your key:

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

Or create `.env` manually -- see [configuration.md](configuration.md).

## Running the Application

### Direct script (no server)

```bash
python test_research_generation.py
```

Runs the full pipeline -- ArXiv search, web search, outline, all sections, PDF export -- and prints progress to stdout.

### API server

```bash
python -m src.main
```

Starts the FastAPI server on `http://localhost:8000` with auto-reload enabled.

### API test client

With the server running in another terminal:

```bash
python test_api_client.py
```

## Testing

### Run the full test suite

```bash
pytest
```

### Run a specific test file

```bash
pytest tests/test_outline_builder.py -v
```

### Test configuration

Test settings are in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

### Test structure

```
tests/
  test_config.py            Settings and environment loading
  test_main.py              Application entry point
  test_server.py            API endpoint routing and validation
  test_paper_agent.py       Agent creation and tool binding
  test_outline_builder.py   Outline generation and validation
  test_section_writer.py    Section content generation
  test_pdf_writer.py        PDF compilation
```

Tests use mocked LLM responses so they run without an API key and without network access.

### Property-based tests

The project uses [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing. Key properties verified:

- Outline structure contains all required sections
- Folder reader categorises every file as processed or skipped
- Citation storage round-trips correctly
- Duplicate citations are deduplicated
- Bibliography contains all added citations
- Paper state serialisation round-trips correctly
- API returns 400 for invalid payloads

## Docker

### Build and run

```bash
docker-compose build
docker-compose up
```

The Dockerfile uses a multi-stage build:

1. **Builder stage** -- installs dependencies into a virtual environment
2. **Runtime stage** -- copies only the venv and source code into a slim Python image

### Docker volumes

The `docker-compose.yml` mounts two volumes:

| Host path | Container path | Purpose |
|-----------|----------------|---------|
| `./references` | `/app/references` | Reference material folder |
| `./output` | `/app/output` | Generated PDF output |

### Environment in Docker

Pass `OPENAI_API_KEY` via the host environment or a `.env` file:

```bash
export OPENAI_API_KEY=sk-proj-...
docker-compose up
```

## Adding a New Tool

1. Create a new file in `src/tools/` (e.g. `src/tools/my_tool.py`).

2. Define an input schema:

```python
from pydantic import BaseModel, Field

class MyToolInput(BaseModel):
    param: str = Field(description="Description of the parameter")
```

3. Implement the tool:

```python
from langchain.tools import BaseTool

class MyTool(BaseTool):
    name: str = "my_tool"
    description: str = "What this tool does"
    args_schema: type[BaseModel] = MyToolInput

    def _run(self, param: str) -> str:
        # Implementation
        return "result"
```

4. Register it in `src/agents/paper_agent.py`:

```python
from src.tools.my_tool import MyTool

# In _create_tools():
my_tool = MyTool()
return [...existing_tools, my_tool]
```

5. Update the `SYSTEM_PROMPT` in `paper_agent.py` to describe the new capability.

6. Add tests in `tests/test_my_tool.py`.

## Code Style

- Type hints on all function signatures
- Docstrings on all public classes and functions
- Pydantic models for all structured data
- Logging via `logging.getLogger(__name__)`
- Error messages that are descriptive and actionable

## Troubleshooting

### `OPENAI_API_KEY not found`

Ensure `.env` exists in the project root and contains a valid key. Restart the server after editing.

### `ModuleNotFoundError`

Run `pip install -e .` to install the package in editable mode.

### ArXiv search fails with PyMuPDF error

The ArXiv tool defaults to metadata-only mode. If you need full document text, install PyMuPDF:

```bash
pip install pymupdf
```

Then set `get_full_documents=True` in `src/tools/arxiv_search.py`.

### Web search returns empty results

DuckDuckGo may rate-limit requests. Wait a few seconds and retry. The system continues working without web search results.

### PDF export fails with missing sections

All seven section types must be generated before exporting. Generate any missing sections first.

### Port 8000 already in use

Change the port in `.env`:

```
API_PORT=8001
```

Or stop the other process using port 8000.
