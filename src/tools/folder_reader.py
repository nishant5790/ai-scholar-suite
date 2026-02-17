"""Folder Reader tool for ingesting reference materials into a vector store."""

import logging
import os
from pathlib import Path
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from src.models.schemas import IngestionResult

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


class FolderReaderInput(BaseModel):
    """Input schema for the FolderReaderTool."""

    folder_path: str = Field(description="Path to the folder containing reference materials")


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text content from a PDF file using PyPDF2."""
    from PyPDF2 import PdfReader

    reader = PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_docx(file_path: str) -> str:
    """Extract text content from a DOCX file using python-docx."""
    from docx import Document

    doc = Document(file_path)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text)


def extract_text_from_plain(file_path: str) -> str:
    """Extract text content from a plain text file (.txt or .md)."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


EXTRACTORS = {
    ".pdf": extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".txt": extract_text_from_plain,
    ".md": extract_text_from_plain,
}


class FolderReaderTool(BaseTool):
    """Tool that reads and indexes reference materials from a folder into a vector store."""

    name: str = "folder_reader"
    description: str = (
        "Reads all supported files (PDF, TXT, DOCX, Markdown) from a specified folder "
        "and indexes their content into the vector store for retrieval by other tools."
    )
    args_schema: Type[BaseModel] = FolderReaderInput

    vector_store: Any = Field(default=None, description="ChromaDB collection for storing document chunks")
    chunk_size: int = Field(default=CHUNK_SIZE, description="Size of text chunks for splitting")
    chunk_overlap: int = Field(default=CHUNK_OVERLAP, description="Overlap between text chunks")

    def _run(self, folder_path: str) -> IngestionResult:
        """Read and index all supported files from the given folder path.

        Args:
            folder_path: Path to the directory containing reference materials.

        Returns:
            IngestionResult with counts of processed/skipped files and total chunks indexed.

        Raises:
            ValueError: If the folder path does not exist or is not a directory.
        """
        path = Path(folder_path)

        if not path.exists():
            raise ValueError(f"Folder path does not exist: {folder_path}")
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")

        files_processed = 0
        files_skipped = 0
        skipped_files: list[str] = []
        total_chunks = 0

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        for entry in sorted(path.iterdir()):
            if not entry.is_file():
                continue

            ext = entry.suffix.lower()

            if ext not in SUPPORTED_EXTENSIONS:
                files_skipped += 1
                skipped_files.append(entry.name)
                logger.warning("Skipping unsupported file format: %s", entry.name)
                continue

            try:
                extractor = EXTRACTORS[ext]
                text = extractor(str(entry))

                if not text.strip():
                    files_processed += 1
                    continue

                chunks = text_splitter.split_text(text)

                if chunks and self.vector_store is not None:
                    ids = [f"{entry.name}_chunk_{i}" for i in range(len(chunks))]
                    metadatas = [
                        {"source": entry.name, "chunk_index": i}
                        for i in range(len(chunks))
                    ]
                    self.vector_store.add(
                        documents=chunks,
                        ids=ids,
                        metadatas=metadatas,
                    )

                total_chunks += len(chunks)
                files_processed += 1

            except Exception as e:
                logger.error("Error processing file %s: %s", entry.name, str(e))
                files_skipped += 1
                skipped_files.append(entry.name)

        return IngestionResult(
            files_processed=files_processed,
            files_skipped=files_skipped,
            skipped_files=skipped_files,
            total_chunks=total_chunks,
        )
