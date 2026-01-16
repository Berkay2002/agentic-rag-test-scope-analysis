from typing import List

from agrag.kg.registry import get_registry


def _invoke_tool(tool, **kwargs) -> str:
    if hasattr(tool, "invoke"):
        return tool.invoke(kwargs)
    if hasattr(tool, "_run"):
        return tool._run(**kwargs)
    raise TypeError("Tool does not support invoke or _run")


def _infer_label_from_id(entity_id: str) -> str:
    registry = get_registry()
    inferred = registry.infer_entity_type(entity_id)
    return inferred or "Function"


def _infer_entity_type_from_id(entity_id: str) -> str | None:
    registry = get_registry()
    return registry.infer_entity_type(entity_id)


def run_fixed_rag(query: str, hybrid_tool, k: int = 10) -> List[str]:
    from agrag.cli.main import _parse_result_ids
    from agrag.config.settings import settings

    registry = get_registry()
    inferred_type = _infer_entity_type_from_id(query)
    entity_type = inferred_type or (registry.normalize_label("TestCase") or "TestCase")
    kwargs = {"query": query, "k": k}
    if entity_type:
        kwargs["entity_type"] = entity_type
    if settings.enable_query_expansion:
        kwargs["enable_query_expansion"] = True
        kwargs["expansion_methods"] = ["synonyms"]
        kwargs["max_expansions"] = settings.max_query_expansions

    result_str = _invoke_tool(hybrid_tool, **kwargs)
    ids = _parse_result_ids(result_str)

    if inferred_type and query not in ids:
        ids.insert(0, query)

    return ids


def run_fixed_graphrag(query: str, hybrid_tool, graph_tool, k: int = 10) -> List[str]:
    from agrag.cli.main import _parse_graph_result_ids

    seed_ids = run_fixed_rag(query=query, hybrid_tool=hybrid_tool, k=k)
    graph_ids: List[str] = []
    registry = get_registry()
    for entity_id in seed_ids[:3]:
        label = _infer_label_from_id(entity_id)
        graph_result = _invoke_tool(
            graph_tool,
            start_node_id=entity_id,
            start_node_label=label,
            relationship_types=[
                r
                for r in [
                    registry.normalize_relationship("TOUCHES"),
                    registry.normalize_relationship("DEFINED_IN"),
                    registry.normalize_relationship("COVERS"),
                    registry.normalize_relationship("VERIFIES"),
                    registry.normalize_relationship("PART_OF"),
                ]
                if r
            ],
            depth=3,
            direction="both",
        )
        graph_ids.extend(_parse_graph_result_ids(graph_result))

    seen = set()
    ordered = []
    for entity_id in seed_ids + graph_ids:
        if entity_id not in seen:
            seen.add(entity_id)
            ordered.append(entity_id)
    return ordered
