"""
File processing service for extracting text from uploaded files.

Supports:
- Text files (.txt, .md)
- Word documents (.docx, .doc)
- PDF files (.pdf)
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FileProcessor:
    """Processes uploaded files and extracts text content."""

    @staticmethod
    async def process_file(file_content: bytes, filename: str) -> dict[str, Any]:
        """
        Process uploaded file and extract text content.

        Args:
            file_content: File content as bytes
            filename: Original filename

        Returns:
            Dictionary with file_name and extracted content

        Raises:
            ValueError: If file type is not supported or processing fails
        """
        file_ext = Path(filename).suffix.lower()

        try:
            if file_ext in ['.txt', '.md']:
                content = file_content.decode('utf-8', errors='replace')
            elif file_ext in ['.docx']:
                content = await FileProcessor._extract_docx(file_content)
            elif file_ext in ['.doc']:
                content = await FileProcessor._extract_doc(file_content)
            elif file_ext == '.pdf':
                content = await FileProcessor._extract_pdf(file_content)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

            return {
                "file_name": filename,
                "content": content,
                "file_type": file_ext,
            }
        except Exception as e:
            logger.error(f"Failed to process file {filename}: {e}")
            raise ValueError(f"Failed to process file: {str(e)}") from e

    @staticmethod
    async def _extract_docx(file_content: bytes) -> str:
        """Extract text from .docx file."""
        try:
            import docx  # python-docx

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name

            try:
                doc = docx.Document(tmp_path)
                paragraphs = [p.text for p in doc.paragraphs]
                return "\n\n".join(paragraphs)
            finally:
                os.unlink(tmp_path)
        except ImportError:
            logger.warning("python-docx not installed")
            raise ValueError(
                "Cannot process .docx files: The 'python-docx' library is not installed.\n\n"
                "To fix this, install the library:\n"
                "1. Open your terminal in the backend directory\n"
                "2. Activate your virtual environment (if using one)\n"
                "3. Run: pip install python-docx\n"
                "4. Restart the backend server"
            )
        except Exception as e:
            raise ValueError(f"Failed to extract DOCX content: {str(e)}") from e

    @staticmethod
    async def _extract_doc(file_content: bytes) -> str:
        """Extract text from .doc file."""
        # .doc files require antiword or similar tools
        # For now, return a message
        logger.warning(".doc files require additional tools (antiword, textract, etc.)")
        return "DOC file format requires additional processing tools. Please convert to DOCX or PDF format."

    @staticmethod
    async def _extract_pdf(file_content: bytes) -> str:
        """Extract text from PDF file."""
        try:
            from pypdf import PdfReader

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name

            try:
                with open(tmp_path, 'rb') as pdf_file:
                    pdf_reader = PdfReader(pdf_file)
                    text_parts = []
                    for page in pdf_reader.pages:
                        text_parts.append(page.extract_text())
                    return "\n\n".join(text_parts)
            finally:
                os.unlink(tmp_path)
        except ImportError:
            logger.warning("pypdf not installed")
            raise ValueError(
                "Cannot process .pdf files: The 'pypdf' library is not installed.\n\n"
                "To fix this, install the library:\n"
                "1. Open your terminal in the backend directory\n"
                "2. Activate your virtual environment (if using one)\n"
                "3. Run: pip install pypdf\n"
                "4. Restart the backend server"
            )
        except Exception as e:
            raise ValueError(f"Failed to extract PDF content: {str(e)}") from e
