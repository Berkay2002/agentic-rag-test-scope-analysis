from agrag.evaluation.baselines.fixed_baselines import run_fixed_graphrag


class _StubHybridTool:
    def _run(self, query: str, k: int = 10):
        return "1. ID: FILE_src_network_handover_py (Score: 0.8)"


class _StubGraphTool:
    def _run(self, start_node_id, start_node_label, relationship_types, depth, direction):
        return (
            "1. Path (depth 2): FILE_src_network_handover_py -> TC_HANDOVER_001\n"
            "   Sequence: File:FILE_src_network_handover_py -> "
            "Function:FUNC_initiate_handover -> TestCase:TC_HANDOVER_001"
        )


def test_fixed_graphrag_combines_retrieval_and_graph() -> None:
    ids = run_fixed_graphrag(
        query="handover changes",
        hybrid_tool=_StubHybridTool(),
        graph_tool=_StubGraphTool(),
        k=5,
    )
    assert "FILE_src_network_handover_py" in ids
    assert "TC_HANDOVER_001" in ids
