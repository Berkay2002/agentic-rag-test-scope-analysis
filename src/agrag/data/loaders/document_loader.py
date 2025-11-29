"""Document loader using Docling for rich document parsing."""

from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
from io import BytesIO

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat, DocumentStream
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
    )
    from docling.chunking import HybridChunker

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

from agrag.data.loaders.base import BaseLoader, Document

logger = logging.getLogger(__name__)


class DoclingDocumentLoader(BaseLoader):
    """
    Document loader using Docling for production-grade parsing.

    Supports: PDF, DOCX, XLSX, PPTX, Markdown, AsciiDoc, HTML, CSV, images, and more.
    Uses AI models (DocLayNet, TableFormer) for accurate layout and table extraction.
    """

    def __init__(
        self,
        file_path: str,
        export_format: str = "markdown",  # "markdown", "text", "json", "html", "doctags"
        use_chunker: bool = False,
        chunker_tokenizer: Optional[str] = None,
        max_num_pages: Optional[int] = None,
        max_file_size: Optional[int] = None,
        artifacts_path: Optional[str] = None,
        table_mode: str = "accurate",  # "accurate" or "fast"
        enable_ocr: bool = True,
        **kwargs,
    ):
        """
        Initialize Docling document loader.

        Args:
            file_path: Path to document (local file or URL)
            export_format: Output format (markdown, text, json, html, doctags)
            use_chunker: Whether to use HybridChunker for semantic chunking
            chunker_tokenizer: Tokenizer for chunker (default: all-MiniLM-L6-v2)
            max_num_pages: Maximum pages to process
            max_file_size: Maximum file size in bytes
            artifacts_path: Path to pre-downloaded models (for offline use)
            table_mode: TableFormer mode ("accurate" or "fast")
            enable_ocr: Enable OCR for images and scanned PDFs
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)

        if not DOCLING_AVAILABLE:
            raise ImportError(
                "Docling is not installed. Install with: pip install langchain-docling docling"
            )

        self.file_path = Path(file_path) if not str(file_path).startswith("http") else file_path
        self.export_format = export_format
        self.use_chunker = use_chunker
        self.chunker_tokenizer = chunker_tokenizer or "sentence-transformers/all-MiniLM-L6-v2"
        self.max_num_pages = max_num_pages
        self.max_file_size = max_file_size
        self.artifacts_path = artifacts_path
        self.table_mode = table_mode
        self.enable_ocr = enable_ocr

        # Initialize converter
        self.converter = self._create_converter()

    def _create_converter(self) -> "DocumentConverter":
        """Create DocumentConverter with configured options."""
        # Configure PDF pipeline options
        pipeline_options = PdfPipelineOptions()

        # Set artifacts path for offline model usage
        if self.artifacts_path:
            pipeline_options.artifacts_path = self.artifacts_path

        # Configure table extraction mode
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.mode = (
            TableFormerMode.ACCURATE if self.table_mode == "accurate" else TableFormerMode.FAST
        )

        # Configure OCR
        pipeline_options.do_ocr = self.enable_ocr

        # Create converter with PDF options
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        return converter

    def load(self) -> List[Document]:
        """
        Load and parse document using Docling.

        Returns:
            List of Document objects with rich metadata
        """
        logger.info(f"Loading document with Docling: {self.file_path}")

        try:
            # Convert document
            result = self.converter.convert(
                self.file_path,
                max_num_pages=self.max_num_pages,
                max_file_size=self.max_file_size,
            )

            docling_doc = result.document

            # Export based on format and chunking preference
            if self.use_chunker:
                return self._load_with_chunker(docling_doc)
            else:
                return self._load_without_chunker(docling_doc)

        except Exception as e:
            logger.error(f"Failed to load document with Docling: {e}")
            raise

    def _load_with_chunker(self, docling_doc) -> List[Document]:
        """Load document with HybridChunker for semantic chunking."""
        chunker = HybridChunker(tokenizer=self.chunker_tokenizer)

        # Chunk the document
        chunk_iter = docling_doc.chunk(chunker=chunker)

        documents = []
        for chunk in chunk_iter:
            # Extract metadata from chunk
            metadata = self._extract_chunk_metadata(chunk, docling_doc)

            # Create Document
            doc = Document(page_content=chunk.text, metadata=metadata)
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} chunks with HybridChunker")
        return documents

    def _load_without_chunker(self, docling_doc) -> List[Document]:
        """Load document and export as single or multiple documents."""
        # Export based on format
        if self.export_format == "markdown":
            content = docling_doc.export_to_markdown()
        elif self.export_format == "text":
            content = docling_doc.export_to_text()
        elif self.export_format == "html":
            content = docling_doc.export_to_html()
        elif self.export_format == "json":
            content = docling_doc.model_dump_json()
        elif self.export_format == "doctags":
            content = docling_doc.export_to_document_tokens()
        else:
            logger.warning(f"Unknown format {self.export_format}, using markdown")
            content = docling_doc.export_to_markdown()

        # Extract global metadata
        metadata = {
            "type": "document",
            "file_path": str(self.file_path),
            "export_format": self.export_format,
            "page_count": len(docling_doc.pages) if hasattr(docling_doc, "pages") else 0,
        }

        # Create single document
        doc = Document(page_content=content, metadata=metadata)

        logger.info(f"Loaded document as single {self.export_format} export")
        return [doc]

    def _extract_chunk_metadata(self, chunk, docling_doc) -> Dict[str, Any]:
        """Extract rich metadata from Docling chunk."""
        metadata = {
            "type": "requirement",  # Default for requirements documents
            "chunk_text": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
        }

        # Extract headings (section hierarchy)
        if hasattr(chunk, "meta") and chunk.meta:
            meta = chunk.meta

            # Get headings for hierarchical context
            if hasattr(meta, "headings") and meta.headings:
                metadata["headings"] = meta.headings
                metadata["title"] = " > ".join(meta.headings)
                metadata["parent_sections"] = meta.headings[:-1] if len(meta.headings) > 1 else []
                metadata["section"] = meta.headings[-1] if meta.headings else ""

            # Get document origin info
            if hasattr(meta, "origin") and meta.origin:
                origin = meta.origin
                metadata["file_path"] = getattr(origin, "filename", str(self.file_path))
                metadata["mimetype"] = getattr(origin, "mimetype", "unknown")

            # Get document items (for page/bbox info)
            if hasattr(meta, "doc_items") and meta.doc_items:
                for item in meta.doc_items:
                    if hasattr(item, "prov") and item.prov:
                        prov = item.prov[0] if isinstance(item.prov, list) else item.prov
                        if hasattr(prov, "page_no"):
                            metadata["page_number"] = prov.page_no
                        if hasattr(prov, "bbox"):
                            bbox = prov.bbox
                            metadata["bbox"] = {
                                "left": getattr(bbox, "l", 0),
                                "top": getattr(bbox, "t", 0),
                                "right": getattr(bbox, "r", 0),
                                "bottom": getattr(bbox, "b", 0),
                            }
                        break

        return metadata

    @classmethod
    def from_binary_stream(
        cls,
        binary_stream: bytes,
        filename: str = "document.pdf",
        **kwargs,
    ) -> "DoclingDocumentLoader":
        """
        Create loader from binary stream instead of file path.

        Args:
            binary_stream: Binary content of document
            filename: Name for the document
            **kwargs: Additional loader arguments

        Returns:
            DoclingDocumentLoader instance
        """
        # Create a temporary loader instance
        loader = cls(file_path=filename, **kwargs)

        # Override the file_path with DocumentStream
        buf = BytesIO(binary_stream)
        loader.file_path = DocumentStream(name=filename, stream=buf)

        return loader


class MultiDocumentLoader(BaseLoader):
    """
    Load multiple documents from a directory using Docling.

    Supports all formats that Docling supports: PDF, DOCX, XLSX, PPTX,
    Markdown, AsciiDoc, HTML, CSV, images, etc.
    """

    def __init__(
        self,
        directory: str,
        formats: Optional[List[str]] = None,
        recursive: bool = True,
        use_chunker: bool = False,
        export_format: str = "markdown",
        **kwargs,
    ):
        """
        Initialize multi-document loader.

        Args:
            directory: Path to directory containing documents
            formats: List of file extensions to include (default: all Docling-supported)
            recursive: Whether to search recursively
            use_chunker: Whether to use semantic chunking
            export_format: Output format for documents
            **kwargs: Additional configuration passed to DoclingDocumentLoader
        """
        super().__init__(**kwargs)

        self.directory = Path(directory)
        self.recursive = recursive
        self.use_chunker = use_chunker
        self.export_format = export_format
        self.loader_kwargs = kwargs

        # Docling-supported formats
        self.default_formats = [
            ".pdf",
            ".docx",
            ".xlsx",
            ".pptx",
            ".md",
            ".markdown",
            ".adoc",
            ".asciidoc",
            ".html",
            ".xhtml",
            ".csv",
            ".png",
            ".jpg",
            ".jpeg",
            ".tiff",
            ".bmp",
            ".webp",
            ".vtt",
        ]

        self.formats = formats or self.default_formats

    def load(self) -> List[Document]:
        """Load all documents from directory."""
        if not self.validate_path(self.directory):
            return []

        logger.info(f"Loading documents from: {self.directory}")

        # Find all matching files
        files = self._find_files()
        logger.info(f"Found {len(files)} documents to process")

        # Load each file
        all_documents = []
        for file_path in files:
            try:
                loader = DoclingDocumentLoader(
                    file_path=str(file_path),
                    export_format=self.export_format,
                    use_chunker=self.use_chunker,
                    **self.loader_kwargs,
                )
                docs = loader.load()
                all_documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        logger.info(f"Loaded {len(all_documents)} chunks from all documents")
        return all_documents

    def _find_files(self) -> List[Path]:
        """Find all documents matching configured formats."""
        files = []

        # Normalize extensions
        extensions = set()
        for fmt in self.formats:
            ext = fmt if fmt.startswith(".") else f".{fmt}"
            extensions.add(ext.lower())

        # Search directory
        pattern = "**/*" if self.recursive else "*"
        for path in self.directory.glob(pattern):
            if not path.is_file():
                continue

            if path.suffix.lower() in extensions:
                files.append(path)

        return files
