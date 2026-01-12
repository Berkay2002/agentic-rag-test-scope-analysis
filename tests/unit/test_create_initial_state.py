from langchain_core.messages import HumanMessage

from agrag.core import create_initial_state


def test_create_initial_state_uses_human_message() -> None:
    state = create_initial_state("hello")
    assert "messages" in state
    assert len(state["messages"]) == 1
    assert isinstance(state["messages"][0], HumanMessage)
    assert state["messages"][0].content == "hello"
