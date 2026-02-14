# Configuration

The application loads configuration from environment variables. A `.env` file in the project root is supported via `python-dotenv`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(empty)* | **Required.** Your OpenAI API key. |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model name used for outline and section generation. |
| `CHROMADB_PATH` | `./chroma_data` | Directory for the persistent ChromaDB vector store. |
| `OUTPUT_DIR` | `./output` | Default directory for generated PDFs. |
| `API_HOST` | `0.0.0.0` | Host the FastAPI server binds to. |
| `API_PORT` | `8000` | Port the FastAPI server listens on. |

## .env File

Create a file named `.env` in the project root:

```
OPENAI_API_KEY=sk-proj-your-key-here
LLM_MODEL=gpt-4o-mini
CHROMADB_PATH=./chroma_data
OUTPUT_DIR=./output
API_HOST=0.0.0.0
API_PORT=8000
```

The file is loaded automatically by `src/config.py` at import time via `load_dotenv()`.

> **Security:** Never commit `.env` to version control. Add it to `.gitignore`.

## Settings Class

All settings are managed through Pydantic's `BaseSettings` in `src/config.py`:

```python
class Settings(BaseSettings):
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    chromadb_path: str = "./chroma_data"
    output_dir: str = "./output"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = {"env_prefix": ""}
```

Environment variables are mapped directly to field names (no prefix). Access settings anywhere in the codebase with:

```python
from src.config import get_settings
settings = get_settings()
print(settings.openai_api_key)
```

## Docker Configuration

When running via Docker, pass the API key as an environment variable:

```bash
docker-compose up
```

The `docker-compose.yml` reads `OPENAI_API_KEY` from your host environment:

```yaml
environment:
  - OPENAI_API_KEY=${OPENAI_API_KEY}
```

Set it in your shell before running:

```bash
export OPENAI_API_KEY=sk-proj-your-key-here
docker-compose up
```

Or use an `.env` file -- docker-compose reads `.env` files in the same directory automatically.

## Model Selection

The `LLM_MODEL` variable controls which OpenAI model is used. Supported values include any model available on your OpenAI account:

| Model | Speed | Cost | Notes |
|-------|-------|------|-------|
| `gpt-4o-mini` | Fast | Low | Default. Good balance for paper generation. |
| `gpt-4o` | Medium | Medium | Higher quality output, better reasoning. |
| `gpt-4` | Slow | High | Maximum quality for complex papers. |

The model is used by both the OutlineBuilder and SectionWriter tools.
