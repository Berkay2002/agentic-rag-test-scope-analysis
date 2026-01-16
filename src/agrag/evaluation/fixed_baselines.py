from typing import List

from agrag.kg.ontology import NodeLabel, RelationshipType


def _invoke_tool(tool, **kwargs) -> str:
    if hasattr(tool, "invoke"):
        return tool.invoke(kwargs)
    if hasattr(tool, "_run"):
        return tool._run(**kwargs)
    raise TypeError("Tool does not support invoke or _run")


def _infer_label_from_id(entity_id: str) -> NodeLabel:
    if entity_id.startswith("CR_"):
        return NodeLabel.CHANGE_REQUEST
    if entity_id.startswith("FILE_"):
        return NodeLabel.FILE
    if entity_id.startswith("COMP_"):
        return NodeLabel.COMPONENT
    if entity_id.startswith("REQ_"):
        return NodeLabel.REQUIREMENT
    if entity_id.startswith("TC_"):
        return NodeLabel.TEST_CASE
    return NodeLabel.FUNCTION


def _infer_entity_type_from_id(entity_id: str) -> str | None:
    if entity_id.startswith("CR_"):
        return NodeLabel.CHANGE_REQUEST.value
    if entity_id.startswith("FILE_"):
        return NodeLabel.FILE.value
    if entity_id.startswith("COMP_"):
        return NodeLabel.COMPONENT.value
    if entity_id.startswith("REQ_"):
        return NodeLabel.REQUIREMENT.value
    if entity_id.startswith("TC_"):
        return NodeLabel.TEST_CASE.value
    if entity_id.startswith("FUNC_"):
        return NodeLabel.FUNCTION.value
    if entity_id.startswith("CLASS_"):
        return NodeLabel.CLASS.value
    if entity_id.startswith("MOD_"):
        return NodeLabel.MODULE.value
    return None


def run_fixed_rag(query: str, hybrid_tool, k: int = 10) -> List[str]:
    from agrag.cli.main import _parse_result_ids

    inferred_type = _infer_entity_type_from_id(query)
    entity_type = inferred_type or NodeLabel.TEST_CASE.value
    kwargs = {"query": query, "k": k}
    if entity_type:
        kwargs["entity_type"] = entity_type

    result_str = _invoke_tool(hybrid_tool, **kwargs)
    ids = _parse_result_ids(result_str)

    if inferred_type and query not in ids:
        ids.insert(0, query)

    return ids


def run_fixed_graphrag(query: str, hybrid_tool, graph_tool, k: int = 10) -> List[str]:
    from agrag.cli.main import _parse_graph_result_ids

    seed_ids = run_fixed_rag(query=query, hybrid_tool=hybrid_tool, k=k)
    graph_ids: List[str] = []
    for entity_id in seed_ids[:3]:
        label = _infer_label_from_id(entity_id)
        graph_result = _invoke_tool(
            graph_tool,
            start_node_id=entity_id,
            start_node_label=label,
            relationship_types=[
                RelationshipType.TOUCHES,
                RelationshipType.DEFINED_IN,
                RelationshipType.COVERS,
                RelationshipType.VERIFIES,
                RelationshipType.PART_OF,
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
