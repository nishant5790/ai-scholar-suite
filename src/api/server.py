"""FastAPI server exposing the AI Research Paper Generator as REST endpoints."""

import logging
import tempfile
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.core.session_manager import SessionManager
from src.core.state_manager import StateManager
from src.models.schemas import (
    CitationStyle,
    ErrorResponse,
    PaperState,
    SectionType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CreateSessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    response: str


class OutlineRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    instructions: str = ""


class SectionRequest(BaseModel):
    feedback: str = ""


class IngestRequest(BaseModel):
    folder_path: str = Field(..., min_length=1)


class BibliographyResponse(BaseModel):
    bibliography: str
    style: str


class ExportPdfRequest(BaseModel):
    output_path: str = Field(..., min_length=1)


class ExportPdfResponse(BaseModel):
    output_path: str


class SaveRequest(BaseModel):
    file_path: str = Field(..., min_length=1)


class LoadRequest(BaseModel):
    file_path: str = Field(..., min_length=1)


class MessageResponse(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(session_manager: Optional[SessionManager] = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        session_manager: Optional pre-configured SessionManager. A new one is
            created when *None*.

    Returns:
        Configured FastAPI app instance.
    """
    app = FastAPI(title="AI Research Paper Generator", version="0.1.0")
    sm = session_manager or SessionManager()
    state_mgr = StateManager()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_session(session_id: str):
        try:
            return sm.get_session(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @app.post("/api/v1/sessions", response_model=CreateSessionResponse, status_code=201)
    def create_session():
        try:
            session_id = sm.create_session()
            return CreateSessionResponse(session_id=session_id)
        except Exception as exc:
            logger.exception("Failed to create session")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/api/v1/sessions/{session_id}/chat", response_model=ChatResponse)
    def chat(session_id: str, request: ChatRequest):
        session = _get_session(session_id)
        try:
            # Placeholder – real agent integration requires an LLM
            return ChatResponse(
                response=f"Received message: {request.message}. Agent integration pending."
            )
        except Exception as exc:
            logger.exception("Chat failed for session %s", session_id)
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/api/v1/sessions/{session_id}/outline")
    def generate_outline(session_id: str, request: OutlineRequest):
        session = _get_session(session_id)
        try:
            from src.tools.outline_builder import OutlineBuilderTool

            # The outline builder needs an LLM; use the session agent's LLM
            # if available, otherwise return an error.
            if session.agent is None:
                raise HTTPException(
                    status_code=500,
                    detail="No agent configured for this session. Cannot generate outline without an LLM.",
                )

            tool = OutlineBuilderTool(
                llm=session.agent,
                vector_store=None,
            )
            outline = tool._run(topic=request.topic, instructions=request.instructions)
            session.paper_state.outline = outline
            session.paper_state.topic = request.topic
            return outline.model_dump()
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.exception("Outline generation failed for session %s", session_id)
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/api/v1/sessions/{session_id}/sections/{section_name}")
    def generate_section(session_id: str, section_name: str, request: SectionRequest):
        session = _get_session(session_id)
        try:
            # Validate section name early
            try:
                SectionType(section_name)
            except ValueError:
                valid = [st.value for st in SectionType]
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid section type: '{section_name}'. Valid types: {', '.join(valid)}",
                )

            if session.agent is None:
                raise HTTPException(
                    status_code=500,
                    detail="No agent configured for this session. Cannot generate section without an LLM.",
                )

            from src.tools.section_writer import SectionWriterTool

            tool = SectionWriterTool(
                llm=session.agent,
                vector_store=None,
                paper_state=session.paper_state,
            )
            content = tool._run(section_name=section_name, feedback=request.feedback)
            session.paper_state.sections[section_name] = content
            return content.model_dump()
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.exception("Section generation failed for session %s", session_id)
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/api/v1/sessions/{session_id}/references/ingest")
    def ingest_references(session_id: str, request: IngestRequest):
        session = _get_session(session_id)
        try:
            from src.tools.folder_reader import FolderReaderTool

            tool = FolderReaderTool(vector_store=None)
            result = tool._run(folder_path=request.folder_path)
            return result.model_dump()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.exception("Reference ingestion failed for session %s", session_id)
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.get("/api/v1/sessions/{session_id}/bibliography", response_model=BibliographyResponse)
    def get_bibliography(session_id: str, style: Optional[str] = None):
        session = _get_session(session_id)
        try:
            from src.tools.reference_manager import ReferenceManagerTool

            bib_style = CitationStyle.APA
            if style:
                try:
                    bib_style = CitationStyle(style)
                except ValueError:
                    valid = [s.value for s in CitationStyle]
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid citation style: '{style}'. Valid styles: {', '.join(valid)}",
                    )

            ref_mgr = ReferenceManagerTool(
                citations=dict(session.paper_state.citations),
                citation_style=bib_style,
            )
            # Rebuild insertion order from paper state
            ref_mgr.insertion_order = list(session.paper_state.citations.keys())
            bibliography = ref_mgr.generate_bibliography(bib_style)
            return BibliographyResponse(bibliography=bibliography, style=bib_style.value)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Bibliography generation failed for session %s", session_id)
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/api/v1/sessions/{session_id}/export/pdf", response_model=ExportPdfResponse)
    def export_pdf(session_id: str, request: ExportPdfRequest):
        session = _get_session(session_id)
        try:
            from src.tools.pdf_writer import PDFWriterTool

            tool = PDFWriterTool(paper_state=session.paper_state)
            result = tool._run(output_path=request.output_path)
            if result.startswith("Error:"):
                raise HTTPException(status_code=400, detail=result)
            return ExportPdfResponse(output_path=result)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("PDF export failed for session %s", session_id)
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/api/v1/sessions/{session_id}/save", response_model=MessageResponse)
    def save_state(session_id: str, request: SaveRequest):
        session = _get_session(session_id)
        try:
            state_mgr.save_state(session.paper_state, request.file_path)
            return MessageResponse(message=f"State saved to {request.file_path}")
        except Exception as exc:
            logger.exception("Save failed for session %s", session_id)
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/api/v1/sessions/{session_id}/load", response_model=MessageResponse)
    def load_state(session_id: str, request: LoadRequest):
        session = _get_session(session_id)
        try:
            loaded = state_mgr.load_state(request.file_path)
            session.paper_state = loaded
            return MessageResponse(message=f"State loaded from {request.file_path}")
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=f"File not found: {request.file_path}")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.exception("Load failed for session %s", session_id)
            raise HTTPException(status_code=500, detail="Internal server error")

    return app


# Default app instance for uvicorn
app = create_app()
