import langgraph.types
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from agrag.core.nodes import execute_tools


def test_execute_tools_hitl_reject(monkeypatch) -> None:
    called = []

    @tool("dummy_tool")
    def dummy_tool(x: int) -> str:
        """Dummy tool for HITL tests."""
        called.append(x)
        return f"ok:{x}"

    def fake_interrupt(payload):
        assert "action_requests" in payload
        assert payload["action_requests"][0]["name"] == "dummy_tool"
        return {"decisions": [{"type": "reject", "message": "nope"}]}

    monkeypatch.setattr(langgraph.types, "interrupt", fake_interrupt)

    state = {
        "messages": [
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[{"name": "dummy_tool", "args": {"x": 1}, "id": "call-1"}],
            ),
        ]
    }

    update = execute_tools(state, tools=[dummy_tool], enable_hitl=True)

    assert called == []
    assert update["tool_call_count"] == 1
    assert len(update["messages"]) == 1
    assert isinstance(update["messages"][0], ToolMessage)
    assert "nope" in update["messages"][0].content


def test_execute_tools_hitl_edit(monkeypatch) -> None:
    called = []

    @tool("dummy_tool")
    def dummy_tool(x: int) -> str:
        """Dummy tool for HITL tests."""
        called.append(x)
        return f"ok:{x}"

    monkeypatch.setattr(
        langgraph.types,
        "interrupt",
        lambda payload: {
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": {"name": "dummy_tool", "args": {"x": 2}},
                }
            ]
        },
    )

    state = {
        "messages": [
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[{"name": "dummy_tool", "args": {"x": 1}, "id": "call-1"}],
            ),
        ]
    }

    update = execute_tools(state, tools=[dummy_tool], enable_hitl=True)

    assert called == [2]
    assert update["tool_call_count"] == 1
    assert len(update["messages"]) == 1
    assert isinstance(update["messages"][0], ToolMessage)
    assert "ok:2" in update["messages"][0].content


def test_execute_tools_tool_not_found() -> None:
    state = {
        "messages": [
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[{"name": "missing_tool", "args": {"x": 1}, "id": "call-1"}],
            ),
        ]
    }

    update = execute_tools(state, tools=[], enable_hitl=False)
    assert update["tool_call_count"] == 1
    assert "not found" in update["messages"][0].content.lower()
