"""Query expansion strategies for improving retrieval recall."""

from typing import List, Dict, Optional
import re
from agrag.config.settings import settings
import logging


class SynonymExpander:
    """Expand queries using synonym dictionaries."""

    # Telecom domain synonyms
    TELECOM_SYNONYMS = {
        "test": ["verification", "validation", "checking", "testcase", "tc"],
        "requirement": ["spec", "specification", "req", "feature", "needs"],
        "function": ["method", "procedure", "routine", "operation"],
        "coverage": ["coverage", "cover", "tested", "test coverage"],
        "verify": ["check", "validate", "test", "confirm", "ensure"],
        "failure": ["error", "bug", "defect", "issue", "problem"],
        "performance": ["speed", "latency", "throughput", "response time"],
        "network": ["net", "connection", "link", "interface"],
        "protocol": ["procedure", "standard", "specification", "interface"],
    }

    def __init__(self, custom_synonyms: Optional[Dict[str, List[str]]] = None):
        """
        Initialize synonym expander.

        Args:
            custom_synonyms: Additional domain-specific synonyms
        """
        self.synonyms = self.TELECOM_SYNONYMS.copy()
        if custom_synonyms:
            for key, values in custom_synonyms.items():
                if key in self.synonyms:
                    self.synonyms[key].extend(values)
                else:
                    self.synonyms[key] = values

    def expand(self, query: str) -> List[str]:
        """Expand query with synonyms."""
        query_lower = query.lower()
        expansions = set()
        expansions.add(query)  # Original query

        # Find matching keywords
        for keyword, synonym_list in self.synonyms.items():
            if keyword in query_lower:
                # Replace keyword with each synonym
                for synonym in synonym_list:
                    expanded = re.sub(
                        r'\b' + re.escape(keyword) + r'\b',
                        synonym,
                        query,
                        flags=re.IGNORECASE
                    )
                    if expanded.lower() != query_lower:
                        expansions.add(expanded)

            # Also add synonyms that contain the original word
            for synonym in synonym_list:
                if keyword in synonym and synonym not in query_lower:
                    # Add variant without modification
                    expansions.add(synonym)

        return list(expansions)


class LLMBasedExpander:
    """Expand queries using language model to generate variations."""

    def __init__(self, llm_service=None):
        """
        Initialize LLM-based expander.

        Args:
            llm_service: LLM service for generating query variations
        """
        self.llm_service = llm_service
        self.prompt_template = """
You are a query expansion assistant for a telecommunications test analysis system.
Given the original query, generate 3-5 alternative ways to ask the same question
that might retrieve different but relevant information.

Original query: "{query}"

Generate alternatives that:
1. Use different terminology for the same concepts
2. Change the focus or perspective
3. Simplify or elaborate
4. Include domain-specific terms

Return each alternative on a new line, starting with "- "
"""

    def expand(self, query: str) -> List[str]:
        """Expand query using LLM-generated variations."""
        if not self.llm_service:
            logging.warning("LLM service not available for query expansion")
            return [query]

        try:
            prompt = self.prompt_template.format(query=query)
            response = self.llm_service.generate(prompt, max_tokens=200)

            expansions = [query]  # Always include original

            # Parse response
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    expansion = line[2:].strip()
                    if expansion and expansion != query:
                        expansions.append(expansion)

            # Limit number of expansions
            return expansions[:settings.max_query_expansions]

        except Exception as e:
            logging.error(f"LLM query expansion failed: {e}")
            return [query]


class PseudoRelevanceExpander:
    """Expand queries using pseudo-relevance feedback."""

    def __init__(self, vector_service, k_initial: int = 10):
        """
        Initialize pseudo-relevance feedback expander.

        Args:
            vector_service: Vector search service for initial retrieval
            k_initial: Number of initial results to analyze
        """
        self.vector_service = vector_service
        self.k_initial = k_initial

    def expand(self, query: str) -> List[str]:
        """
        Expand query using terms from top initial results.

        Args:
            query: Original query

        Returns:
            List of expanded queries
        """
        try:
            # Get initial results
            query_embedding = self.vector_service.embed_query(query)
            initial_results = self.vector_service.search(query_embedding, k=self.k_initial)

            if not initial_results:
                return [query]

            # Extract key terms from top results
            expanded_terms = self._extract_key_terms(initial_results, query)

            # Generate expanded queries
            expansions = [query]

            # Add query with expanded terms
            if expanded_terms:
                expanded_query = f"{query} {' '.join(expanded_terms[:5])}"
                expansions.append(expanded_query)

            return expansions[:settings.max_query_expansions]

        except Exception as e:
            logging.error(f"Pseudo-relevance expansion failed: {e}")
            return [query]

    def _extract_key_terms(self, results, original_query: str) -> List[str]:
        """Extract important terms from results."""
        from collections import Counter
        import re

        original_terms = set(original_query.lower().split())
        all_terms = []

        for result in results[:3]:  # Use top 3 results
            # Extract words, filtering for length and relevance
            words = re.findall(r'\b[a-zA-Z]{4,}\b', result.content.lower())
            all_terms.extend(words)

        # Count term frequencies
        term_counts = Counter(all_terms)

        # Filter: terms not in original query, reasonably frequent
        key_terms = []
        for term, count in term_counts.most_common():
            if term not in original_terms and count >= 2 and term not in key_terms:
                key_terms.append(term)
            if len(key_terms) >= 10:
                break

        return key_terms


class QueryExpansionService:
    """Main service for query expansion with multiple strategies."""

    def __init__(
        self,
        llm_service=None,
        vector_service=None,
        custom_synonyms: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize query expansion service.

        Args:
            llm_service: LLM service for expansion
            vector_service: Vector service for pseudo-relevance
            custom_synonyms: Custom synonym dictionary
        """
        self.strategies = {
            "synonyms": SynonymExpander(custom_synonyms),
        }

        if llm_service:
            self.strategies["llm"] = LLMBasedExpander(llm_service)

        if vector_service:
            self.strategies["pseudo_relevance"] = PseudoRelevanceExpander(vector_service)

    def expand(
        self,
        query: str,
        methods: List[str] = None,
        max_expansions: int = None
    ) -> List[str]:
        """
        Expand query using specified methods.

        Args:
            query: Original query
            methods: List of expansion methods to use
            max_expansions: Maximum number of expansions to return

        Returns:
            List of expanded queries (including original)
        """
        if not methods:
            methods = ["synonyms"]  # Default

        all_expansions = set()
        all_expansions.add(query)

        for method in methods:
            if method in self.strategies:
                try:
                    expansions = self.strategies[method].expand(query)
                    all_expansions.update(expansions)
                except Exception as e:
                    logging.error(f"Query expansion failed for method {method}: {e}")

        # Always return original query first
        result = [query]
        max_exp = max_expansions or settings.max_query_expansions

        # Add other expansions
        for expansion in list(all_expansions)[1:max_exp]:
            if expansion != query:
                result.append(expansion)

        return result
