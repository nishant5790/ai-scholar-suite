"""Unit tests for the FolderReaderTool."""

import pytest

from src.models.schemas import IngestionResult
from src.tools.folder_reader import (
    SUPPORTED_EXTENSIONS,
    FolderReaderTool,
    extract_text_from_plain,
)


@pytest.fixture
def chroma_collection(request):
    """Create an in-memory ChromaDB collection for testing."""
    import uuid

    import chromadb

    client = chromadb.Client()
    collection_name = f"test_{uuid.uuid4().hex[:12]}"
    collection = client.get_or_create_collection(name=collection_name)
    yield collection
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass


@pytest.fixture
def folder_reader(chroma_collection):
    """Create a FolderReaderTool with a test vector store."""
    return FolderReaderTool(vector_store=chroma_collection)


class TestFolderReaderValidation:
    def test_nonexistent_path_raises_error(self, folder_reader):
        with pytest.raises(ValueError, match="does not exist"):
            folder_reader._run("/nonexistent/path/to/folder")

    def test_file_path_raises_error(self, folder_reader, tmp_path):
        file_path = tmp_path / "somefile.txt"
        file_path.write_text("content")
        with pytest.raises(ValueError, match="not a directory"):
            folder_reader._run(str(file_path))

    def test_empty_directory(self, folder_reader, tmp_path):
        result = folder_reader._run(str(tmp_path))
        assert result.files_processed == 0
        assert result.files_skipped == 0
        assert result.skipped_files == []
        assert result.total_chunks == 0


class TestTextFileIngestion:
    def test_single_txt_file(self, folder_reader, tmp_path):
        (tmp_path / "notes.txt").write_text("Some research notes about AI.")
        result = folder_reader._run(str(tmp_path))
        assert result.files_processed == 1
        assert result.files_skipped == 0
        assert result.total_chunks >= 1

    def test_single_md_file(self, folder_reader, tmp_path):
        (tmp_path / "readme.md").write_text("# Research\n\nThis is a markdown file.")
        result = folder_reader._run(str(tmp_path))
        assert result.files_processed == 1
        assert result.files_skipped == 0
        assert result.total_chunks >= 1

    def test_empty_text_file_processed_with_zero_chunks(self, folder_reader, tmp_path):
        (tmp_path / "empty.txt").write_text("")
        result = folder_reader._run(str(tmp_path))
        assert result.files_processed == 1
        assert result.total_chunks == 0


class TestUnsupportedFiles:
    def test_unsupported_extension_skipped(self, folder_reader, tmp_path):
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        result = folder_reader._run(str(tmp_path))
        assert result.files_processed == 0
        assert result.files_skipped == 1
        assert "image.png" in result.skipped_files

    def test_mixed_supported_and_unsupported(self, folder_reader, tmp_path):
        (tmp_path / "paper.txt").write_text("Research content here.")
        (tmp_path / "data.csv").write_text("a,b,c")
        (tmp_path / "notes.md").write_text("# Notes")
        (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8")
        result = folder_reader._run(str(tmp_path))
        assert result.files_processed == 2
        assert result.files_skipped == 2
        assert set(result.skipped_files) == {"data.csv", "photo.jpg"}


class TestVectorStoreIndexing:
    def test_chunks_indexed_into_vector_store(self, chroma_collection, tmp_path):
        (tmp_path / "doc.txt").write_text("Machine learning is a subset of artificial intelligence.")
        reader = FolderReaderTool(vector_store=chroma_collection)
        result = reader._run(str(tmp_path))
        assert result.total_chunks >= 1
        # Verify content is queryable in the vector store
        query_result = chroma_collection.query(
            query_texts=["machine learning"],
            n_results=1,
        )
        assert len(query_result["documents"][0]) >= 1

    def test_metadata_includes_source_filename(self, chroma_collection, tmp_path):
        (tmp_path / "ref.txt").write_text("Important reference material for the study.")
        reader = FolderReaderTool(vector_store=chroma_collection)
        reader._run(str(tmp_path))
        query_result = chroma_collection.query(
            query_texts=["reference material"],
            n_results=1,
        )
        assert query_result["metadatas"][0][0]["source"] == "ref.txt"

    def test_no_vector_store_still_counts_chunks(self, tmp_path):
        (tmp_path / "doc.txt").write_text("Some text content.")
        reader = FolderReaderTool(vector_store=None)
        result = reader._run(str(tmp_path))
        assert result.files_processed == 1
        assert result.total_chunks >= 1


class TestDocxIngestion:
    def test_docx_file_processed(self, folder_reader, tmp_path):
        from docx import Document

        doc = Document()
        doc.add_paragraph("This is a test DOCX document with research content.")
        docx_path = tmp_path / "paper.docx"
        doc.save(str(docx_path))
        result = folder_reader._run(str(tmp_path))
        assert result.files_processed == 1
        assert result.total_chunks >= 1


class TestPdfIngestion:
    def test_pdf_file_processed(self, folder_reader, tmp_path):
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        pdf_path = tmp_path / "paper.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        c.drawString(72, 700, "This is a test PDF document with research content.")
        c.save()
        result = folder_reader._run(str(tmp_path))
        assert result.files_processed == 1
        assert result.total_chunks >= 1


class TestIngestionResult:
    def test_result_is_ingestion_result_model(self, folder_reader, tmp_path):
        (tmp_path / "test.txt").write_text("content")
        result = folder_reader._run(str(tmp_path))
        assert isinstance(result, IngestionResult)

    def test_counts_add_up(self, folder_reader, tmp_path):
        (tmp_path / "a.txt").write_text("text a")
        (tmp_path / "b.md").write_text("text b")
        (tmp_path / "c.csv").write_text("1,2,3")
        (tmp_path / "d.exe").write_bytes(b"\x00")
        result = folder_reader._run(str(tmp_path))
        total_files = len(list(tmp_path.iterdir()))
        assert result.files_processed + result.files_skipped == total_files


class TestExtractTextFromPlain:
    def test_reads_utf8_content(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello, world!", encoding="utf-8")
        assert extract_text_from_plain(str(f)) == "Hello, world!"
