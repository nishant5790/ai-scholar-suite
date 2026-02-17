"""Application entry point.

Initializes the FastAPI app with SessionManager, ChromaDB, and CORS middleware.
"""

import chromadb
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from src.api.server import create_app
from src.config import get_settings
from src.core.session_manager import SessionManager


def build_app():
    """Build and configure the full application stack.

    Creates a ChromaDB client/collection, a SessionManager, and the FastAPI app
    with CORS middleware enabled for development.
    """
    settings = get_settings()

    # ChromaDB persistent client and collection
    chroma_client = chromadb.PersistentClient(path=settings.chromadb_path)
    chroma_collection = chroma_client.get_or_create_collection(name="reference_materials")

    # Session manager
    session_manager = SessionManager()

    # FastAPI app
    app = create_app(session_manager=session_manager)

    # CORS middleware – allow all origins for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach shared resources to app state for access in endpoints
    app.state.chroma_client = chroma_client
    app.state.chroma_collection = chroma_collection
    app.state.settings = settings

    return app


app = build_app()


def main():
    """Run the application with uvicorn."""
    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )


if __name__ == "__main__":
    main()
