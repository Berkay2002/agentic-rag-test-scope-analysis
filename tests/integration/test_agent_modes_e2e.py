from typing import Dict, List

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agrag.core import create_agent_graph, create_initial_state


class _FakeLLM:
    def __init__(self, tool_name: str, tool_args: Dict, final_answer: str = "Done") -> None:
        self._tool_name = tool_name
        self._tool_args = tool_args
        self._final_answer = final_answer
        self._call_count = 0

    def bind_tools(self, tools):
        self._tools = tools
        return self

    def invoke(self, messages):
        self._call_count += 1
        if self._call_count == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": self._tool_name,
                        "args": self._tool_args,
                    }
                ],
            )
        return AIMessage(content=self._final_answer)


def _tool_messages(messages: List) -> List[ToolMessage]:
    return [message for message in messages if isinstance(message, ToolMessage)]


def _first_requirement_id(dataset: Dict) -> str:
    for entity in dataset.get("entities", []):
        entity_id = entity.get("id", "")
        if entity_id.startswith("REQ_"):
            return entity_id
    raise AssertionError("No requirement ID found in dataset")


def test_yolo_mode_executes_tools(monkeypatch, postgres_client, neo4j_client, dataset) -> None:
    requirement_id = _first_requirement_id(dataset)
    fake_llm = _FakeLLM(
        tool_name="keyword_search",
        tool_args={"query": requirement_id, "k": 3, "entity_type": "Requirement"},
    )

    import agrag.core.nodes as nodes

    monkeypatch.setattr(nodes, "get_llm", lambda: fake_llm)

    graph = create_agent_graph(
        checkpointer=None,
        enable_hitl=False,
        postgres_client=postgres_client,
        neo4j_client=neo4j_client,
    )

    final_state = graph.invoke(create_initial_state("Find requirement"))
    tool_messages = _tool_messages(final_state.get("messages", []))

    assert tool_messages
    assert any("Keyword Search Results" in msg.content for msg in tool_messages)


def test_safe_mode_requires_approval(monkeypatch, postgres_client, neo4j_client, dataset) -> None:
    requirement_id = _first_requirement_id(dataset)
    fake_llm = _FakeLLM(
        tool_name="keyword_search",
        tool_args={"query": requirement_id, "k": 3, "entity_type": "Requirement"},
    )

    import agrag.core.nodes as nodes

    monkeypatch.setattr(nodes, "get_llm", lambda: fake_llm)

    checkpointer = MemorySaver()
    graph = create_agent_graph(
        checkpointer=checkpointer,
        enable_hitl=True,
        postgres_client=postgres_client,
        neo4j_client=neo4j_client,
    )

    config = {"configurable": {"thread_id": "e2e-hitl"}}
    initial_state = create_initial_state("Find requirement")

    interrupt_event = None
    for event in graph.stream(initial_state, config=config, stream_mode="values"):
        if "__interrupt__" in event:
            interrupt_event = event
            break

    assert interrupt_event is not None

    command = Command(resume={"decisions": [{"type": "approve"}]})
    for _event in graph.stream(command, config=config, stream_mode="values"):
        pass

    final_state = graph.get_state(config).values
    tool_messages = _tool_messages(final_state.get("messages", []))

    assert tool_messages
    assert any("Keyword Search Results" in msg.content for msg in tool_messages)
