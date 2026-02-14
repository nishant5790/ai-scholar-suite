# Architecture

## Overview

The AI Research Paper Generator follows an **agent-tool pattern**. A central LangChain agent receives user requests and delegates work to specialised tools. The system is exposed as a REST API and maintains per-session state so multiple papers can be written concurrently.

```
Client (HTTP)
  |
  v
FastAPI Server  -->  Session Manager  -->  Paper Agent (LangChain)
                                               |
                         +---------------------+---------------------+
                         |           |         |         |           |
                    OutlineBuilder  SectionWriter  FolderReader  WebSearch
                         |           |                               |
                    PDFWriter   ReferenceManager              ArxivSearch
                         |           |
                         v           v
                   ChromaDB     Conversation Memory
```

## Component Responsibilities

### FastAPI Server (`src/api/server.py`)

- Validates incoming HTTP requests with Pydantic models
- Resolves sessions via `SessionManager`
- Routes requests to the appropriate tool or agent
- Returns JSON responses; maps errors to proper HTTP status codes

### Session Manager (`src/core/session_manager.py`)

- Stores `PaperSession` objects keyed by a UUID session ID
- Each session holds its own `PaperState` and an optional `AgentExecutor`
- Sessions live in-memory; persistence is handled by the State Manager

### Paper Agent (`src/agents/paper_agent.py`)

- Creates a `ChatOpenAI` instance (model configurable via `LLM_MODEL`)
- Instantiates all seven tools and binds them to a LangChain agent
- Uses `MemorySaver` for conversation history across invocations
- System prompt describes available capabilities so the LLM can route correctly

### Tools

Each tool extends `langchain.tools.BaseTool` with a typed `args_schema` (Pydantic model) and implements `_run()`.

| Tool | File | Needs LLM | Needs Vector Store |
|------|------|-----------|--------------------|
| OutlineBuilder | `outline_builder.py` | Yes | Optional |
| SectionWriter | `section_writer.py` | Yes | Optional |
| FolderReader | `folder_reader.py` | No | Yes |
| ReferenceManager | `reference_manager.py` | No | No |
| PDFWriter | `pdf_writer.py` | No | No |
| WebSearch | `web_search.py` | No | No |
| ArxivSearch | `arxiv_search.py` | No | No |

### ChromaDB Vector Store

- Persistent client stored at `./chroma_data` (configurable)
- Collection `reference_materials` holds chunked document embeddings
- Queried by OutlineBuilder and SectionWriter for context-aware generation
- Populated by FolderReader when the user ingests a reference folder

### State Manager (`src/core/state_manager.py`)

- Serialises `PaperState` to JSON and writes to disk
- Loads and deserialises JSON back into `PaperState`
- Enables save/resume across sessions

## Data Flow

### End-to-End Paper Generation

1. **Session creation** -- client calls `POST /api/v1/sessions`. A new `PaperSession` is created with an empty `PaperState`.

2. **Research** (optional) -- the agent's WebSearch and ArxivSearch tools gather external material. Results inform subsequent tool calls.

3. **Reference ingestion** (optional) -- `POST /api/v1/sessions/{id}/references/ingest` triggers FolderReader, which reads supported files, chunks them, and stores embeddings in ChromaDB.

4. **Outline generation** -- `POST /api/v1/sessions/{id}/outline` invokes OutlineBuilder. The LLM generates a structured JSON outline containing all seven standard sections. The outline is stored in `PaperState.outline`.

5. **Section writing** -- for each section type, `POST /api/v1/sessions/{id}/sections/{name}` invokes SectionWriter. The tool receives the outline and any previously generated sections as context. Output is stored in `PaperState.sections`.

6. **PDF export** -- `POST /api/v1/sessions/{id}/export/pdf` invokes PDFWriter. It compiles the title page, all sections, and bibliography into a formatted PDF.

7. **State persistence** -- `POST /api/v1/sessions/{id}/save` serialises the current `PaperState` to a JSON file. `POST /api/v1/sessions/{id}/load` restores it.

## Data Models

All models are defined in `src/models/schemas.py` using Pydantic.

### PaperState

The central state object for a paper in progress:

```python
class PaperState(BaseModel):
    title: str = ""
    author: str = ""
    topic: str = ""
    outline: Optional[PaperOutline] = None
    sections: dict[str, SectionContent] = {}
    citations: dict[str, CitationMetadata] = {}
    citation_style: CitationStyle = CitationStyle.APA
```

### PaperOutline

```python
class PaperOutline(BaseModel):
    topic: str
    sections: list[OutlineSection]

class OutlineSection(BaseModel):
    section_type: SectionType
    title: str
    key_points: list[str]
    subsections: list[OutlineSection] = []
```

### SectionContent

```python
class SectionContent(BaseModel):
    section_type: SectionType
    title: str
    content: str
    citations: list[str] = []
```

### SectionType (enum)

`abstract`, `introduction`, `literature_review`, `methodology`, `results`, `discussion`, `conclusion`

### CitationStyle (enum)

`apa`, `ieee`, `mla`
