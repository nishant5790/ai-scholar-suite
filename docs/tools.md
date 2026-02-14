# Tools Reference

Every tool inherits from `langchain.tools.BaseTool`, declares an `args_schema` (Pydantic model for input validation), and implements `_run()`. This document covers each tool's purpose, inputs, outputs, and usage examples.

---

## 1. Outline Builder

**File:** `src/tools/outline_builder.py`

Generates a structured research paper outline from a topic string.

### Input

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `topic` | str | Yes | The research topic |
| `instructions` | str | No | Additional constraints or focus areas |

### Output

Returns a `PaperOutline` containing a list of `OutlineSection` objects. Each section has a `section_type`, `title`, `key_points`, and optional `subsections`.

The tool guarantees all seven standard sections are present: abstract, introduction, literature_review, methodology, results, discussion, conclusion. If the LLM omits any, the tool fills them in with defaults.

### How It Works

1. Queries the ChromaDB vector store (if available) for relevant reference context.
2. Builds a prompt asking the LLM to produce a JSON outline.
3. Parses the JSON response, fills in missing sections, and sorts them in standard order.
4. Validates that every section has a non-empty title and at least one key point.

### Example

```python
from src.tools.outline_builder import OutlineBuilderTool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
tool = OutlineBuilderTool(llm=llm, vector_store=None)
outline = tool._run(
    topic="AI fine tuning",
    instructions="Focus on LoRA, QLoRA, and prompt tuning techniques."
)
for section in outline.sections:
    print(f"{section.section_type.value}: {section.title}")
```

---

## 2. Section Writer

**File:** `src/tools/section_writer.py`

Generates academic content for a single paper section.

### Input

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `section_name` | str | Yes | One of the seven `SectionType` values |
| `feedback` | str | No | Revision instructions for an existing section |

### Output

Returns a `SectionContent` object with `section_type`, `title`, `content` (the prose), and `citations` (list of citation IDs referenced).

### How It Works

1. Validates the section name against the `SectionType` enum.
2. Queries ChromaDB for reference context related to the topic and section.
3. Builds a prompt that includes the paper outline, previously generated sections, reference materials, and any feedback.
4. Parses the LLM's JSON response into `SectionContent`.

The tool maintains consistency across sections by including all previously written sections in each subsequent prompt.

### Example

```python
from src.tools.section_writer import SectionWriterTool
from src.models.schemas import PaperState

tool = SectionWriterTool(llm=llm, vector_store=None, paper_state=paper_state)
abstract = tool._run(section_name="abstract")
print(abstract.content)
```

---

## 3. Web Search (DuckDuckGo)

**File:** `src/tools/web_search.py`

Searches the web using the DuckDuckGo API for recent articles, blog posts, and general information.

### Input

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | str | Yes | -- | Search query |
| `max_results` | int | No | 5 | Number of results (1--10) |

### Output

Returns a `WebSearchResult` with:

- `query` -- the original query
- `results` -- list of dicts, each with `title`, `url`, and `snippet`
- `total_results` -- count of results returned

### Example

```python
from src.tools.web_search import WebSearchTool

tool = WebSearchTool()
results = tool._run(query="transformer fine tuning techniques 2026", max_results=5)
for r in results.results:
    print(f"{r['title']}: {r['url']}")
```

---

## 4. ArXiv Search

**File:** `src/tools/arxiv_search.py`

Retrieves academic papers from ArXiv using `langchain_community.retrievers.ArxivRetriever`.

### Input

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | str | Yes | -- | Search query for ArXiv |
| `max_docs` | int | No | 2 | Number of papers to retrieve (1--10) |

### Output

Returns an `ArxivSearchResult` with:

- `query` -- the original query
- `papers` -- list of `ArxivPaper` objects
- `total_papers` -- count

Each `ArxivPaper` contains: `title`, `authors`, `published` (date string), `arxiv_id`, `summary`, `pdf_url`, `entry_id`.

### Notes

- By default, `get_full_documents` is set to `False` (metadata and summaries only). Set it to `True` in the code if `pymupdf` is installed and you want full PDF text extraction.
- The `published` field is automatically converted to a string regardless of whether ArXiv returns a `date` object or string.

### Example

```python
from src.tools.arxiv_search import ArxivSearchTool

tool = ArxivSearchTool()
results = tool._run(query="LoRA low rank adaptation", max_docs=3)
for paper in results.papers:
    print(f"[{paper.published}] {paper.title} -- {paper.authors}")
```

---

## 5. Folder Reader

**File:** `src/tools/folder_reader.py`

Reads files from a local directory and stores their content as embeddings in ChromaDB.

### Input

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `folder_path` | str | Yes | Path to a directory containing reference files |

### Output

Returns an `IngestionResult` with: `files_processed`, `files_skipped`, `skipped_files` (list of names), `total_chunks`.

### Supported Formats

`.pdf`, `.txt`, `.docx`, `.md`

Files with other extensions are skipped and logged.

---

## 6. Reference Manager

**File:** `src/tools/reference_manager.py`

Manages citations and generates formatted bibliographies.

### Key Methods

- **`add_citation(metadata: CitationMetadata) -> str`** -- stores a citation and returns its ID. Duplicate citations (same author + title + year) are merged.
- **`generate_bibliography(style: CitationStyle) -> str`** -- produces a formatted reference list.

### Supported Styles

| Style | Example |
|-------|---------|
| APA | Author, A. (2024). Title. *Source*. |
| IEEE | [1] A. Author, "Title," *Source*, 2024. |
| MLA | Author, A. "Title." *Source*, 2024. |

### In-Text Markers

The tool produces markers that correspond to bibliography entries:

- APA: `(Author, 2024)`
- IEEE: `[1]`
- MLA: `(Author)`

---

## 7. PDF Writer

**File:** `src/tools/pdf_writer.py`

Compiles a complete `PaperState` into a formatted PDF using ReportLab.

### Input

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `output_path` | str | Yes | File path for the generated PDF |

### Output

Returns the output file path as a string on success, or an error string prefixed with `"Error:"`.

### Requirements

All seven sections must be present in `PaperState.sections` before export. If any are missing, the tool returns an error listing the absent sections.

### PDF Contents

- Title page with paper title, author, and date
- Each section with formatted headings and body text
- Bibliography at the end (if citations exist)
- Consistent fonts, margins, and page numbers throughout
