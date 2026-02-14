# AI Research Paper Generator

An AI-powered research paper generator using LangChain agents. This system helps researchers write structured, high-quality academic papers by leveraging multiple AI tools including web search, ArXiv retrieval, and intelligent section writing.

## Features

- **Intelligent Paper Outline Generation**: Automatically generates structured outlines for research papers
- **Section-by-Section Writing**: AI-powered writing for each paper section (abstract, introduction, methodology, etc.)
- **Web Search Integration**: Search the web using DuckDuckGo for the latest information and articles
- **ArXiv Integration**: Search and retrieve academic papers and preprints from ArXiv
- **Reference Management**: Automatic citation management with support for APA, IEEE, and MLA styles
- **PDF Export**: Export completed papers as professionally formatted PDFs
- **REST API**: Full-featured API for integration with other applications
- **Session Management**: Maintain multiple paper writing sessions concurrently

## Tools Available

1. **Outline Builder**: Generates structured paper outlines based on topic and instructions
2. **Section Writer**: Writes individual sections with academic tone and proper structure
3. **Folder Reader**: Ingests reference materials from local folders (PDF, TXT, DOCX, MD)
4. **Reference Manager**: Manages citations and generates bibliographies
5. **PDF Writer**: Exports papers to formatted PDF documents
6. **Web Search**: Searches DuckDuckGo for recent information and web content
7. **ArXiv Search**: Retrieves academic papers from ArXiv database

## Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Internet connection (for web search and ArXiv retrieval)

## Installation

### 1. Clone or Navigate to the Repository

```bash
cd /Users/nkumar/AI-research
```

### 2. Install Dependencies

Using pip:

```bash
pip install -e .
```

Or install with development dependencies:

```bash
pip install -e ".[dev]"
```

### 3. Configure Environment Variables

Create a `.env` file in the project root (already created with your API key):

```bash
# .env file
OPENAI_API_KEY=your-openai-api-key-here
LLM_MODEL=gpt-4o-mini
CHROMADB_PATH=./chroma_data
OUTPUT_DIR=./output
API_HOST=0.0.0.0
API_PORT=8000
```

The `.env` file has already been created with your OpenAI API key.

## Quick Start

### Option 1: Direct Test Script (Fastest)

Run the test script to generate a research paper on "AI fine tuning":

```bash
python test_research_generation.py
```

This will:
- Search ArXiv for academic papers on AI fine tuning
- Search the web for latest information
- Generate a paper outline
- Write multiple sections (abstract, introduction, methodology, conclusion)
- Export the paper as PDF to `./output/ai_fine_tuning_research.pdf`

### Option 2: API Server

#### Start the Server

```bash
python -m src.main
```

The server will start on `http://localhost:8000`

#### Test the API

In a separate terminal, run the API test client:

```bash
python test_api_client.py
```

## Usage Examples

### Using the Tools Directly

```python
from src.tools.web_search import WebSearchTool
from src.tools.arxiv_search import ArxivSearchTool
from src.tools.outline_builder import OutlineBuilderTool
from langchain_openai import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Search the web
web_tool = WebSearchTool()
web_results = web_tool._run(query="AI fine tuning 2026", max_results=5)
print(f"Found {web_results.total_results} web results")

# Search ArXiv
arxiv_tool = ArxivSearchTool()
arxiv_results = arxiv_tool._run(query="AI fine tuning", max_docs=3)
print(f"Found {arxiv_results.total_papers} papers")

# Generate outline
outline_tool = OutlineBuilderTool(llm=llm, vector_store=None)
outline = outline_tool._run(
    topic="AI fine tuning",
    instructions="Create a comprehensive survey paper outline"
)
```

### Using the REST API

#### Create a Session

```bash
curl -X POST http://localhost:8000/api/v1/sessions
```

Response:
```json
{
  "session_id": "abc123..."
}
```

#### Generate Outline

```bash
curl -X POST http://localhost:8000/api/v1/sessions/{session_id}/outline \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "AI fine tuning",
    "instructions": "Create a comprehensive survey paper outline"
  }'
```

#### Generate a Section

```bash
curl -X POST http://localhost:8000/api/v1/sessions/{session_id}/sections/abstract \
  -H "Content-Type: application/json" \
  -d '{
    "feedback": ""
  }'
```

#### Export to PDF

```bash
curl -X POST http://localhost:8000/api/v1/sessions/{session_id}/export/pdf \
  -H "Content-Type: application/json" \
  -d '{
    "output_path": "./output/my_paper.pdf"
  }'
```

## API Endpoints

- `POST /api/v1/sessions` - Create a new paper session
- `POST /api/v1/sessions/{id}/chat` - Send a message to the agent
- `POST /api/v1/sessions/{id}/outline` - Generate paper outline
- `POST /api/v1/sessions/{id}/sections/{name}` - Generate a specific section
- `POST /api/v1/sessions/{id}/references/ingest` - Ingest reference folder
- `GET /api/v1/sessions/{id}/bibliography` - Get formatted bibliography
- `POST /api/v1/sessions/{id}/export/pdf` - Export paper as PDF
- `POST /api/v1/sessions/{id}/save` - Save paper state
- `POST /api/v1/sessions/{id}/load` - Load paper state

## Section Types

Valid section types for paper generation:

- `abstract` - Paper abstract/summary
- `introduction` - Introduction section
- `literature_review` - Literature review
- `methodology` - Methodology/methods section
- `results` - Results section
- `discussion` - Discussion section
- `conclusion` - Conclusion section

## Citation Styles

Supported citation formatting styles:

- `apa` - APA (American Psychological Association)
- `ieee` - IEEE
- `mla` - MLA (Modern Language Association)

## Project Structure

```
AI-research/
├── src/
│   ├── agents/
│   │   └── paper_agent.py          # Main agent orchestrator
│   ├── api/
│   │   └── server.py                # FastAPI REST API
│   ├── core/
│   │   ├── session_manager.py       # Session management
│   │   └── state_manager.py         # Paper state serialization
│   ├── models/
│   │   └── schemas.py               # Pydantic data models
│   ├── tools/
│   │   ├── arxiv_search.py          # ArXiv retrieval tool
│   │   ├── folder_reader.py         # Local file ingestion
│   │   ├── outline_builder.py       # Outline generation
│   │   ├── pdf_writer.py            # PDF export
│   │   ├── reference_manager.py     # Citation management
│   │   ├── section_writer.py        # Section writing
│   │   └── web_search.py            # DuckDuckGo search tool
│   ├── config.py                    # Configuration settings
│   └── main.py                      # Application entry point
├── tests/                           # Unit tests
├── .env                             # Environment variables (API keys)
├── pyproject.toml                   # Project dependencies
├── test_research_generation.py     # Direct test script
├── test_api_client.py              # API integration test
└── README.md                        # This file
```

## Output

Generated papers are saved in the `./output/` directory by default. This can be configured via the `OUTPUT_DIR` environment variable.

## Troubleshooting

### API Key Issues

If you see "OPENAI_API_KEY not found" errors:

1. Ensure the `.env` file exists in the project root
2. Verify the `OPENAI_API_KEY` is set correctly in `.env`
3. Restart the server or test script after updating `.env`

### Import Errors

If you encounter import errors:

```bash
pip install -e .
```

This installs the package in editable mode with all dependencies.

### Connection Errors

If the API test fails to connect:

1. Make sure the server is running: `python -m src.main`
2. Check that port 8000 is not in use by another application
3. Verify the server started successfully (check console output)

### Web Search Issues

If DuckDuckGo search fails:

- This may be due to rate limiting or network issues
- The system will continue to work without web search results
- Try again after a short delay

## Development

### Running Tests

```bash
pytest tests/
```

### Running with Docker

```bash
# Build the image
docker-compose build

# Start the service
docker-compose up
```

The API will be available at `http://localhost:8000`

## License

This project is part of an AI research initiative.

## Contributing

When adding new tools or features:

1. Follow the existing tool pattern (inherit from `BaseTool`)
2. Add proper input validation with Pydantic models
3. Include comprehensive docstrings
4. Add unit tests in the `tests/` directory
5. Update this README with new features

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the test scripts for usage examples
3. Examine the existing tools for implementation patterns

## Acknowledgments

Built with:
- [LangChain](https://python.langchain.com/) - Agent framework
- [OpenAI](https://openai.com/) - Language models
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [ReportLab](https://www.reportlab.com/) - PDF generation
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/) - Web search
- [ArXiv](https://arxiv.org/) - Academic paper repository
