"""AST-based code splitter for structure-aware chunking."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser, Node

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

from agrag.data.loaders.base import BaseTextSplitter, Document

logger = logging.getLogger(__name__)


class CodeSplitter(BaseTextSplitter):
    """
    AST-based code splitter using tree-sitter.

    Extracts semantic units (functions, classes, methods) from source code
    while preserving parent-child relationships and structural context.
    """

    def __init__(
        self,
        language: str = "python",
        extract_functions: bool = True,
        extract_classes: bool = True,
        extract_methods: bool = True,
        include_docstrings: bool = True,
        include_decorators: bool = True,
        chunk_size: int = 2048,
        **kwargs,
    ):
        """
        Initialize the code splitter.

        Args:
            language: Programming language (currently supports 'python')
            extract_functions: Extract standalone functions
            extract_classes: Extract class definitions
            extract_methods: Extract class methods
            include_docstrings: Include docstrings in chunks
            include_decorators: Include decorators in chunks
            chunk_size: Maximum chunk size (soft limit)
            **kwargs: Additional configuration
        """
        super().__init__(chunk_size=chunk_size, **kwargs)

        self.language = language
        self.extract_functions = extract_functions
        self.extract_classes = extract_classes
        self.extract_methods = extract_methods
        self.include_docstrings = include_docstrings
        self.include_decorators = include_decorators

        if not TREE_SITTER_AVAILABLE:
            raise ImportError(
                "tree-sitter and tree-sitter-python are required for CodeSplitter. "
                "Install with: pip install tree-sitter tree-sitter-python"
            )

        # Initialize tree-sitter parser
        self.parser = Parser()
        if language == "python":
            PY_LANGUAGE = Language(tspython.language())
            self.parser.set_language(PY_LANGUAGE)
        else:
            raise ValueError(f"Unsupported language: {language}")

    def split_text(self, text: str) -> List[str]:
        """
        Split code text into semantic chunks.

        Args:
            text: Source code text

        Returns:
            List of code chunks
        """
        chunks = self.split_code(text)
        return [chunk["content"] for chunk in chunks]

    def split_code(
        self,
        code: str,
        file_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Split code into semantic chunks with metadata.

        Args:
            code: Source code string
            file_path: Optional file path for metadata

        Returns:
            List of chunk dictionaries with content and metadata
        """
        chunks = []

        # Parse code into AST
        tree = self.parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node

        # Extract different types of code units
        if self.extract_functions:
            chunks.extend(self._extract_functions(code, root_node, file_path))

        if self.extract_classes:
            chunks.extend(self._extract_classes(code, root_node, file_path))

        # If no chunks extracted, return whole file as single chunk
        if not chunks:
            chunks.append(
                {
                    "content": code,
                    "metadata": {
                        "type": "module",
                        "file_path": file_path,
                        "line_start": 1,
                        "line_end": len(code.split("\n")),
                    },
                }
            )

        return chunks

    def _extract_functions(
        self,
        code: str,
        root_node: Node,
        file_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Extract standalone function definitions."""
        functions = []

        # Query for function definitions at module level
        query_string = """
        (function_definition
            name: (identifier) @func_name) @func_def
        """

        # Find all function nodes
        func_nodes = self._find_nodes_by_type(root_node, "function_definition")

        for func_node in func_nodes:
            # Skip methods (functions inside classes)
            if self._is_inside_class(func_node):
                continue

            chunk = self._extract_function_chunk(code, func_node, file_path)
            if chunk:
                functions.append(chunk)

        return functions

    def _extract_classes(
        self,
        code: str,
        root_node: Node,
        file_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Extract class definitions and optionally their methods."""
        classes = []

        # Find all class nodes
        class_nodes = self._find_nodes_by_type(root_node, "class_definition")

        for class_node in class_nodes:
            # Extract the whole class
            class_chunk = self._extract_class_chunk(code, class_node, file_path)
            if class_chunk:
                classes.append(class_chunk)

            # Extract individual methods if configured
            if self.extract_methods:
                method_chunks = self._extract_methods_from_class(
                    code, class_node, file_path, class_chunk["metadata"].get("name")
                )
                classes.extend(method_chunks)

        return classes

    def _extract_function_chunk(
        self,
        code: str,
        func_node: Node,
        file_path: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Extract a single function as a chunk."""
        lines = code.split("\n")

        # Get function name
        name_node = self._find_child_by_type(func_node, "identifier")
        func_name = self._get_node_text(code, name_node) if name_node else "unknown"

        # Get function signature
        params_node = self._find_child_by_type(func_node, "parameters")
        signature = (
            f"def {func_name}{self._get_node_text(code, params_node) if params_node else '()'}"
        )

        # Get return type if present
        return_type = self._find_child_by_type(func_node, "type")
        if return_type:
            signature += f" -> {self._get_node_text(code, return_type)}"

        # Get docstring
        docstring = self._extract_docstring(code, func_node)

        # Get decorators
        decorators = []
        if self.include_decorators:
            decorators = self._extract_decorators(code, func_node)

        # Extract full function code
        start_line = func_node.start_point[0]
        end_line = func_node.end_point[0]

        # Include decorators in the content
        if decorators:
            decorator_lines = [lines[dec["line"]] for dec in decorators if dec["line"] < start_line]
            if decorator_lines:
                start_line = min(dec["line"] for dec in decorators)

        content = "\n".join(lines[start_line : end_line + 1])

        return {
            "content": content,
            "metadata": {
                "type": "function",
                "name": func_name,
                "signature": signature,
                "file_path": file_path,
                "line_start": start_line + 1,  # 1-indexed
                "line_end": end_line + 1,
                "docstring": docstring,
                "decorators": [d["name"] for d in decorators],
                "parent_class": None,
            },
        }

    def _extract_class_chunk(
        self,
        code: str,
        class_node: Node,
        file_path: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Extract a class definition as a chunk."""
        lines = code.split("\n")

        # Get class name
        name_node = self._find_child_by_type(class_node, "identifier")
        class_name = self._get_node_text(code, name_node) if name_node else "unknown"

        # Get base classes
        base_classes = []
        arg_list = self._find_child_by_type(class_node, "argument_list")
        if arg_list:
            for child in arg_list.children:
                if child.type == "identifier":
                    base_classes.append(self._get_node_text(code, child))

        # Get class docstring
        docstring = self._extract_docstring(code, class_node)

        # Get decorators
        decorators = []
        if self.include_decorators:
            decorators = self._extract_decorators(code, class_node)

        # Extract methods
        method_nodes = self._find_nodes_by_type(class_node, "function_definition")
        method_names = []
        for method_node in method_nodes:
            name_node = self._find_child_by_type(method_node, "identifier")
            if name_node:
                method_names.append(self._get_node_text(code, name_node))

        # Extract full class code
        start_line = class_node.start_point[0]
        end_line = class_node.end_point[0]

        # Include decorators in the content
        if decorators:
            decorator_lines = [lines[dec["line"]] for dec in decorators if dec["line"] < start_line]
            if decorator_lines:
                start_line = min(dec["line"] for dec in decorators)

        content = "\n".join(lines[start_line : end_line + 1])

        return {
            "content": content,
            "metadata": {
                "type": "class",
                "name": class_name,
                "file_path": file_path,
                "line_start": start_line + 1,
                "line_end": end_line + 1,
                "docstring": docstring,
                "base_classes": base_classes,
                "methods": method_names,
                "decorators": [d["name"] for d in decorators],
            },
        }

    def _extract_methods_from_class(
        self,
        code: str,
        class_node: Node,
        file_path: Optional[str] = None,
        class_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Extract individual methods from a class."""
        methods = []

        method_nodes = self._find_nodes_by_type(class_node, "function_definition")

        for method_node in method_nodes:
            chunk = self._extract_function_chunk(code, method_node, file_path)
            if chunk:
                # Update metadata to indicate it's a method
                chunk["metadata"]["type"] = "method"
                chunk["metadata"]["parent_class"] = class_name
                methods.append(chunk)

        return methods

    def _extract_docstring(self, code: str, node: Node) -> Optional[str]:
        """Extract docstring from a function or class node."""
        if not self.include_docstrings:
            return None

        # Look for expression_statement with string as first child
        body = self._find_child_by_type(node, "block")
        if not body:
            return None

        for child in body.children:
            if child.type == "expression_statement":
                string_node = self._find_child_by_type(child, "string")
                if string_node:
                    docstring = self._get_node_text(code, string_node)
                    # Remove quotes
                    docstring = docstring.strip('"""').strip("'''").strip('"').strip("'")
                    return docstring.strip()

        return None

    def _extract_decorators(self, code: str, node: Node) -> List[Dict[str, Any]]:
        """Extract decorators from a function or class node."""
        decorators = []

        # Look backwards for decorator nodes
        prev_node = node.prev_sibling
        while prev_node and prev_node.type == "decorator":
            decorator_text = self._get_node_text(code, prev_node)
            decorators.insert(
                0,
                {
                    "name": decorator_text.lstrip("@").strip(),
                    "line": prev_node.start_point[0],
                },
            )
            prev_node = prev_node.prev_sibling

        return decorators

    def _find_nodes_by_type(self, root: Node, node_type: str) -> List[Node]:
        """Recursively find all nodes of a specific type."""
        nodes = []

        def traverse(node: Node):
            if node.type == node_type:
                nodes.append(node)
            for child in node.children:
                traverse(child)

        traverse(root)
        return nodes

    def _find_child_by_type(self, node: Node, child_type: str) -> Optional[Node]:
        """Find first direct child of a specific type."""
        for child in node.children:
            if child.type == child_type:
                return child
        return None

    def _is_inside_class(self, node: Node) -> bool:
        """Check if a node is inside a class definition."""
        parent = node.parent
        while parent:
            if parent.type == "class_definition":
                return True
            parent = parent.parent
        return False

    def _get_node_text(self, code: str, node: Node) -> str:
        """Extract text from a node."""
        if not node:
            return ""
        return code[node.start_byte : node.end_byte]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split code documents into semantic chunks.

        Args:
            documents: List of Document objects containing code

        Returns:
            List of split Document objects with metadata
        """
        split_docs = []

        for doc in documents:
            file_path = doc.metadata.get("file_path") or doc.metadata.get("source")

            # Split the code
            chunks = self.split_code(doc.page_content, file_path)

            # Convert to Document objects
            for chunk in chunks:
                metadata = doc.metadata.copy()
                metadata.update(chunk["metadata"])
                split_docs.append(Document(page_content=chunk["content"], metadata=metadata))

        return split_docs
