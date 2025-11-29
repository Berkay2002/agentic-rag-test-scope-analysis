"""Document loaders for ingesting code and documentation."""

from agrag.data.loaders.base import BaseLoader, BaseTextSplitter, Document
from agrag.data.loaders.code_loader import CodeLoader
from agrag.data.loaders.document_loader import DoclingDocumentLoader, MultiDocumentLoader
from agrag.data.loaders.splitters import CodeSplitter, MarkdownSplitter, SemanticSplitter
from agrag.data.loaders.tgf_loader import TGFCSVLoader, TGFTestRecord

__all__ = [
    "BaseLoader",
    "BaseTextSplitter",
    "Document",
    "CodeLoader",
    "DoclingDocumentLoader",
    "MultiDocumentLoader",
    "CodeSplitter",
    "MarkdownSplitter",
    "SemanticSplitter",
    "TGFCSVLoader",
    "TGFTestRecord",
]
