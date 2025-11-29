"""Header-based document splitter for hierarchical documents."""

from typing import List, Dict, Any, Optional
import re
import logging

from agrag.data.loaders.base import BaseTextSplitter, Document

logger = logging.getLogger(__name__)


class MarkdownSplitter(BaseTextSplitter):
    """
    Header-based document splitter for Markdown/reStructuredText.

    Splits documents by hierarchical headers while preserving parent context.
    Ensures that each chunk maintains its position in the document hierarchy.
    """

    def __init__(
        self,
        headers_to_split_on: Optional[List[tuple]] = None,
        strip_headers: bool = False,
        return_each_section: bool = True,
        chunk_size: int = 1024,
        **kwargs,
    ):
        """
        Initialize the markdown splitter.

        Args:
            headers_to_split_on: List of (header_symbol, header_name) tuples
                Default: [("#", "h1"), ("##", "h2"), ("###", "h3")]
            strip_headers: Whether to remove headers from chunks
            return_each_section: Return individual sections vs combined chunks
            chunk_size: Maximum chunk size
            **kwargs: Additional configuration
        """
        super().__init__(chunk_size=chunk_size, **kwargs)

        self.headers_to_split_on = headers_to_split_on or [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ]
        self.strip_headers = strip_headers
        self.return_each_section = return_each_section

    def split_text(self, text: str) -> List[str]:
        """
        Split markdown text by headers.

        Args:
            text: Markdown text to split

        Returns:
            List of text chunks
        """
        chunks = self.split_markdown(text)
        return [chunk["content"] for chunk in chunks]

    def split_markdown(
        self,
        text: str,
        file_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Split markdown text into semantic chunks with metadata.

        Args:
            text: Markdown text
            file_path: Optional file path for metadata

        Returns:
            List of chunk dictionaries with content and metadata
        """
        lines = text.split("\n")
        chunks = []

        # Track hierarchy
        current_headers = {}  # level -> (title, line_num)
        current_content = []
        current_start_line = 1

        for line_num, line in enumerate(lines, 1):
            header_match = self._parse_header(line)

            if header_match:
                # Save previous section if it has content
                if current_content:
                    chunk = self._create_chunk(
                        current_content,
                        current_headers.copy(),
                        current_start_line,
                        line_num - 1,
                        file_path,
                    )
                    if chunk:
                        chunks.append(chunk)
                    current_content = []

                # Update hierarchy
                level, title = header_match
                current_headers[level] = (title, line_num)

                # Remove lower-level headers
                keys_to_remove = [k for k in current_headers.keys() if k > level]
                for k in keys_to_remove:
                    del current_headers[k]

                current_start_line = line_num

                # Include header in content unless stripping
                if not self.strip_headers:
                    current_content.append(line)
            else:
                current_content.append(line)

        # Add final section
        if current_content:
            chunk = self._create_chunk(
                current_content,
                current_headers.copy(),
                current_start_line,
                len(lines),
                file_path,
            )
            if chunk:
                chunks.append(chunk)

        return chunks

    def _parse_header(self, line: str) -> Optional[tuple]:
        """
        Parse a line to detect markdown headers.

        Args:
            line: Line of text

        Returns:
            Tuple of (level, title) or None
        """
        line = line.strip()

        # Check ATX-style headers (# Header)
        for symbol, header_name in self.headers_to_split_on:
            if line.startswith(symbol + " "):
                level = len(symbol)
                title = line[len(symbol) :].strip()
                return (level, title)

        # Check for numbered sections (1.2.3 Title)
        numbered_pattern = r"^(\d+(?:\.\d+)*)\.?\s+(.+)$"
        match = re.match(numbered_pattern, line)
        if match:
            section_num = match.group(1)
            title = match.group(2)
            level = section_num.count(".") + 1
            return (level, f"{section_num} {title}")

        return None

    def _create_chunk(
        self,
        content_lines: List[str],
        headers: Dict[int, tuple],
        start_line: int,
        end_line: int,
        file_path: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Create a chunk with hierarchical metadata.

        Args:
            content_lines: Lines of content
            headers: Dictionary of level -> (title, line_num)
            start_line: Starting line number
            end_line: Ending line number
            file_path: File path

        Returns:
            Chunk dictionary or None if empty
        """
        content = "\n".join(content_lines).strip()
        if not content:
            return None

        # Build parent sections chain
        sorted_levels = sorted(headers.keys())
        parent_sections = []
        current_section = None
        section_level = None

        for level in sorted_levels:
            title, _ = headers[level]
            if current_section is None:
                current_section = title
                section_level = level
            else:
                parent_sections.append(title)

        # Extract section number if present
        section_num = None
        if current_section:
            num_match = re.match(r"^(\d+(?:\.\d+)*)", current_section)
            if num_match:
                section_num = num_match.group(1)

        return {
            "content": content,
            "metadata": {
                "type": "requirement",  # Assume requirements for now
                "section": section_num,
                "title": current_section,
                "parent_sections": parent_sections,
                "level": section_level or 1,
                "file_path": file_path,
                "line_start": start_line,
                "line_end": end_line,
            },
        }

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split markdown documents into hierarchical chunks.

        Args:
            documents: List of Document objects

        Returns:
            List of split Document objects with hierarchy metadata
        """
        split_docs = []

        for doc in documents:
            file_path = doc.metadata.get("file_path") or doc.metadata.get("source")

            # Split the markdown
            chunks = self.split_markdown(doc.page_content, file_path)

            # Convert to Document objects
            for chunk in chunks:
                metadata = doc.metadata.copy()
                metadata.update(chunk["metadata"])
                split_docs.append(Document(page_content=chunk["content"], metadata=metadata))

        return split_docs
