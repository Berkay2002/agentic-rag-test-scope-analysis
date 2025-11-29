# Docling Integration

This project now uses **Docling** for production-grade document parsing, replacing basic PDF/DOCX parsers with IBM Research's AI-powered solution.

## What is Docling?

Docling is an MIT-licensed document conversion library that uses specialized AI models:
- **DocLayNet**: Accurate layout analysis and page element detection
- **TableFormer**: State-of-the-art table structure recognition

## Supported Formats

### Input Formats
- **Documents**: PDF, DOCX, XLSX, PPTX
- **Markup**: Markdown, AsciiDoc, HTML, XHTML
- **Data**: CSV
- **Images**: PNG, JPEG, TIFF, BMP, WEBP
- **Video**: WebVTT (subtitles)
- **Schema-specific**: USPTO XML, JATS XML, Docling JSON

### Output Formats
- **Markdown**: Structured with headers (default)
- **Text**: Plain text without markers
- **JSON**: Lossless Docling Document serialization
- **HTML**: With image embedding or referencing
- **Doctags**: Rich markup with layout characteristics

## Installation

```bash
# Install Docling dependencies
poetry install

# The system will auto-download models (~500MB) on first use
# Models are cached in $HOME/.cache/docling/models
```

### Offline/Air-Gapped Usage

For offline environments, prefetch models:

```bash
# Download all models
docling-tools models download

# Or specify a custom path
docling-tools models download --output /path/to/models

# Then use with CLI
agrag load docs /path/to/docs --artifacts-path /path/to/models
```

## Usage

### Basic Document Loading

```bash
# Load all supported documents from a directory
agrag load docs /path/to/requirements

# Specify formats (default: pdf,docx,markdown)
agrag load docs /path/to/docs --formats pdf,docx,xlsx,pptx

# Use semantic chunking (default: enabled)
agrag load docs /path/to/docs --use-chunker

# Disable chunking (whole document export)
agrag load docs /path/to/docs --no-chunker
```

### Advanced Options

```bash
# Fast table extraction (lower quality, faster)
agrag load docs /path/to/docs --table-mode fast

# Limit pages per document
agrag load docs /path/to/docs --max-pages 50

# Export as different format
agrag load docs /path/to/docs --export-format json
agrag load docs /path/to/docs --export-format html
```

### Programmatic Usage

```python
from agrag.data.loaders import DoclingDocumentLoader

# Load a single PDF with chunking
loader = DoclingDocumentLoader(
    file_path="/path/to/requirements.pdf",
    use_chunker=True,
    table_mode="accurate",
    max_num_pages=100,
)
documents = loader.load()

# Load from URL
loader = DoclingDocumentLoader(
    file_path="https://arxiv.org/pdf/2408.09869",
    use_chunker=True,
)
documents = loader.load()

# Load from binary stream
with open("document.pdf", "rb") as f:
    binary_data = f.read()

loader = DoclingDocumentLoader.from_binary_stream(
    binary_stream=binary_data,
    filename="document.pdf",
)
documents = loader.load()

# Batch loading
from agrag.data.loaders import MultiDocumentLoader

loader = MultiDocumentLoader(
    directory="/path/to/docs",
    recursive=True,
    use_chunker=True,
    export_format="markdown",
)
all_docs = loader.load()
```

## Features

### Rich Metadata Extraction

Docling extracts comprehensive metadata:

```python
doc.metadata = {
    "type": "requirement",
    "headings": ["1. Introduction", "1.1 Motivation"],
    "title": "1. Introduction > 1.1 Motivation",
    "parent_sections": ["1. Introduction"],
    "section": "1.1 Motivation",
    "page_number": 3,
    "bbox": {
        "left": 108.0,
        "top": 405.14,
        "right": 504.00,
        "bottom": 330.78
    },
    "file_path": "requirements.pdf",
    "mimetype": "application/pdf"
}
```

### Semantic Chunking

When `use_chunker=True`, Docling's `HybridChunker` creates semantically coherent chunks:
- Respects document structure (sections, paragraphs)
- Maintains context boundaries
- Optimized for RAG workflows
- Token-aware chunking with configurable limits

### Table Extraction

Two modes available:

```python
# Accurate mode (default): slower, better quality
loader = DoclingDocumentLoader(
    file_path="report.pdf",
    table_mode="accurate"
)

# Fast mode: faster processing, acceptable quality
loader = DoclingDocumentLoader(
    file_path="report.pdf",
    table_mode="fast"
)
```

## Architecture

The document loading pipeline:

```
Documents (PDF, DOCX, etc.)
    ↓
Docling Converter
    ├── Layout Analysis (DocLayNet)
    ├── Table Recognition (TableFormer)
    └── OCR (optional)
    ↓
Docling Document
    ├── HybridChunker (if enabled)
    └── Export (Markdown/Text/JSON/HTML)
    ↓
AgRAG Documents
    ↓
Dual Storage Writer
    ├── Neo4j (graph structure)
    ├── PostgreSQL (vector embeddings)
    └── BM25 (keyword search)
```

## Comparison: Docling vs Basic Parsers

| Feature | pypdf/python-docx | Docling |
|---------|------------------|---------|
| Layout analysis | ❌ | ✅ AI-powered (DocLayNet) |
| Table extraction | ❌ Basic | ✅ Advanced (TableFormer) |
| Semantic chunking | ❌ | ✅ HybridChunker |
| Rich metadata | ❌ | ✅ Bounding boxes, hierarchy |
| Multi-format | ❌ Limited | ✅ 15+ formats |
| OCR support | ❌ | ✅ Built-in |
| Production ready | ❌ | ✅ IBM Research |

## Performance Tips

### GPU Acceleration

For best performance, use GPU acceleration when available:

```bash
# Check if GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# Docling will automatically use GPU if available
```

### Resource Limits

Control CPU usage:

```bash
# Limit to 2 threads (default: 4)
export OMP_NUM_THREADS=2
agrag load docs /path/to/docs
```

### File Size Limits

Prevent loading extremely large files:

```python
from agrag.data.loaders import DoclingDocumentLoader

loader = DoclingDocumentLoader(
    file_path="huge_document.pdf",
    max_num_pages=100,        # Process only first 100 pages
    max_file_size=20971520,   # 20MB limit
)
```

## Troubleshooting

### "Token indices sequence length is longer..."

This warning can be safely ignored. It occurs during model initialization and doesn't affect conversion quality. [More info](https://github.com/DS4SD/docling-core/issues/119#issuecomment-2577418826)

### Models not downloading

Manually download models:

```bash
docling-tools models download
```

### Out of memory errors

Reduce resource usage:

```bash
# Limit pages
agrag load docs /path/to/docs --max-pages 50

# Use fast table mode
agrag load docs /path/to/docs --table-mode fast

# Disable chunking
agrag load docs /path/to/docs --no-chunker
```

## What Still Uses Custom Loaders?

**Code repositories** still use tree-sitter AST parsing:

```bash
# Code loading (Python, Java, etc.)
agrag load repo /path/to/code --languages python,java
```

Docling focuses on documents, not source code, so AST-based code parsing remains essential for analyzing test coverage and dependencies.

## References

- [Docling GitHub](https://github.com/docling-project/docling)
- [Docling Documentation](https://docling-project.github.io/docling/)
- [LangChain Docling Integration](https://github.com/docling-project/docling-langchain)
- [DocLayNet Paper](https://arxiv.org/abs/2206.01062)
- [TableFormer Paper](https://arxiv.org/abs/2203.01017)
- [Docling Technical Report](https://arxiv.org/abs/2408.09869)
