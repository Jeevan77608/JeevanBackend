"""Tests for FileProcessor: text, markdown, docx, doc, pdf, unsupported."""

from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

from src.services.file_processor import FileProcessor


# =============================================================================
# Text / Markdown Files
# =============================================================================


class TestTextFiles:
    """Tests for processing plain text and markdown files."""

    async def test_process_text_file(self) -> None:
        """Process a .txt file; content is decoded UTF-8."""
        content = b"Hello, world!"
        result = await FileProcessor.process_file(content, "readme.txt")
        assert result["file_name"] == "readme.txt"
        assert result["content"] == "Hello, world!"
        assert result["file_type"] == ".txt"

    async def test_process_markdown_file(self) -> None:
        """Process a .md file; content is decoded UTF-8."""
        content = b"# Heading\n\nSome paragraph."
        result = await FileProcessor.process_file(content, "notes.md")
        assert result["file_name"] == "notes.md"
        assert result["content"] == "# Heading\n\nSome paragraph."
        assert result["file_type"] == ".md"

    async def test_process_text_file_with_special_chars(self) -> None:
        """UTF-8 content with unicode characters is preserved."""
        text = "Caf\u00e9 \u2014 na\u00efve r\u00e9sum\u00e9 \u2603"
        content = text.encode("utf-8")
        result = await FileProcessor.process_file(content, "unicode.txt")
        assert result["content"] == text

    async def test_process_text_file_with_invalid_utf8(self) -> None:
        """Invalid UTF-8 bytes are replaced rather than raising."""
        content = b"Hello \xff\xfe world"
        result = await FileProcessor.process_file(content, "bad.txt")
        # errors='replace' should insert replacement characters
        assert "Hello" in result["content"]
        assert "world" in result["content"]


# =============================================================================
# DOC Files
# =============================================================================


class TestDocFiles:
    """Tests for .doc file processing (legacy format)."""

    async def test_process_doc_file(self) -> None:
        """Process a .doc file; returns conversion message."""
        result = await FileProcessor.process_file(b"binary-doc-data", "old.doc")
        assert result["file_name"] == "old.doc"
        assert "convert" in result["content"].lower() or "DOCX" in result["content"]
        assert result["file_type"] == ".doc"


# =============================================================================
# DOCX Files (library not installed)
# =============================================================================


class TestDocxFiles:
    """Tests for .docx file processing when python-docx is unavailable."""

    async def test_process_docx_file_no_library(self) -> None:
        """When python-docx is not installed, raises ValueError with install instructions."""
        real_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "docx":
                raise ImportError("No module named 'docx'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ValueError, match="python-docx"):
                await FileProcessor.process_file(b"fake-docx", "test.docx")


# =============================================================================
# PDF Files (library not installed)
# =============================================================================


class TestPdfFiles:
    """Tests for .pdf file processing when pypdf is unavailable."""

    async def test_process_pdf_file_no_library(self) -> None:
        """When pypdf is not installed, raises ValueError with install instructions."""
        real_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "pypdf":
                raise ImportError("No module named 'pypdf'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ValueError, match="pypdf"):
                await FileProcessor.process_file(b"fake-pdf", "test.pdf")


# =============================================================================
# Unsupported Files
# =============================================================================


class TestUnsupportedFiles:
    """Tests for unsupported file types."""

    async def test_process_unsupported_file(self) -> None:
        """Processing an .exe file raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            await FileProcessor.process_file(b"\x00\x01", "malware.exe")

    async def test_process_unsupported_file_unknown_extension(self) -> None:
        """Processing a file with an unknown extension raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            await FileProcessor.process_file(b"data", "archive.7z")
