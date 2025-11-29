"""Code repository loader for ingesting source code."""

from typing import List, Optional, Set
from pathlib import Path
import logging

from agrag.data.loaders.base import BaseLoader, Document
from agrag.data.loaders.splitters.code_splitter import CodeSplitter

logger = logging.getLogger(__name__)


class CodeLoader(BaseLoader):
    """
    Load and parse source code from a repository.

    Walks directory tree, detects file types, and uses AST-based
    splitting to extract semantic code units.
    """

    def __init__(
        self,
        repo_path: str,
        languages: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        respect_gitignore: bool = True,
        **kwargs,
    ):
        """
        Initialize the code loader.

        Args:
            repo_path: Path to the code repository
            languages: List of languages to process (default: ["python"])
            include_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude
            respect_gitignore: Whether to respect .gitignore rules
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)

        self.repo_path = Path(repo_path)
        self.languages = languages or ["python"]
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or [
            "**/__pycache__/**",
            "**/*.pyc",
            "**/.git/**",
            "**/node_modules/**",
            "**/venv/**",
            "**/.venv/**",
            "**/build/**",
            "**/dist/**",
        ]
        self.respect_gitignore = respect_gitignore

        # Language to file extension mapping
        self.lang_extensions = {
            "python": [".py"],
            "java": [".java"],
            "javascript": [".js", ".jsx"],
            "typescript": [".ts", ".tsx"],
            "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".h"],
            "c": [".c", ".h"],
        }

        # Initialize splitters per language
        self.splitters = {}
        for lang in self.languages:
            if lang == "python":
                self.splitters[lang] = CodeSplitter(language=lang)

    def load(self) -> List[Document]:
        """
        Load all code files from the repository.

        Returns:
            List of Document objects with code chunks and metadata
        """
        if not self.validate_path(self.repo_path):
            return []

        logger.info(f"Loading code from repository: {self.repo_path}")

        # Find all matching files
        files = self._find_files()
        logger.info(f"Found {len(files)} code files to process")

        # Load and split each file
        all_documents = []
        for file_path in files:
            try:
                docs = self._load_file(file_path)
                all_documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        logger.info(f"Loaded {len(all_documents)} code chunks from repository")
        return all_documents

    def _find_files(self) -> List[Path]:
        """
        Find all code files matching the configured patterns.

        Returns:
            List of file paths
        """
        files = []

        # Get all extensions we're looking for
        extensions = set()
        for lang in self.languages:
            extensions.update(self.lang_extensions.get(lang, []))

        # Walk the repository
        for path in self.repo_path.rglob("*"):
            if not path.is_file():
                continue

            # Check extension
            if path.suffix not in extensions:
                continue

            # Check exclude patterns
            if self._is_excluded(path):
                continue

            # Check include patterns (if specified)
            if self.include_patterns and not self._is_included(path):
                continue

            files.append(path)

        return files

    def _is_excluded(self, path: Path) -> bool:
        """Check if a path matches any exclude pattern."""
        relative_path = path.relative_to(self.repo_path)

        for pattern in self.exclude_patterns:
            if relative_path.match(pattern):
                return True

        return False

    def _is_included(self, path: Path) -> bool:
        """Check if a path matches any include pattern."""
        if not self.include_patterns:
            return True

        relative_path = path.relative_to(self.repo_path)

        for pattern in self.include_patterns:
            if relative_path.match(pattern):
                return True

        return False

    def _load_file(self, file_path: Path) -> List[Document]:
        """
        Load and split a single code file.

        Args:
            file_path: Path to the code file

        Returns:
            List of Document objects for code chunks
        """
        # Read file
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return []

        # Determine language
        lang = self._detect_language(file_path)
        if not lang or lang not in self.splitters:
            logger.warning(f"No splitter available for language: {lang}")
            # Return whole file as single document
            return [
                Document(
                    page_content=code,
                    metadata={
                        "file_path": str(file_path.relative_to(self.repo_path)),
                        "language": lang,
                        "type": "file",
                    },
                )
            ]

        # Split using appropriate splitter
        splitter = self.splitters[lang]
        chunks = splitter.split_code(code, str(file_path.relative_to(self.repo_path)))

        # Convert to Document objects
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["content"],
                metadata={
                    **chunk["metadata"],
                    "language": lang,
                    "repository": str(self.repo_path),
                },
            )
            documents.append(doc)

        return documents

    def _detect_language(self, file_path: Path) -> Optional[str]:
        """
        Detect programming language from file extension.

        Args:
            file_path: Path to the file

        Returns:
            Language name or None
        """
        suffix = file_path.suffix

        for lang, extensions in self.lang_extensions.items():
            if suffix in extensions:
                return lang

        return None
