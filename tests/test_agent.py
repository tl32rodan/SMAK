import pytest

from smak.agent.react import ReActAgent
from smak.agent.router import ToolRouter
from smak.agent.tools import Tool, ToolRegistry


def test_tool_registry_registers_and_lists() -> None:
    registry = ToolRegistry()
    tool = Tool(name="echo", description="", handler=lambda value: value)

    registry.register(tool)

    assert registry.get("echo") == tool
    assert registry.list_tools() == [tool]


def test_tool_router_uses_default_tool() -> None:
    registry = ToolRegistry()
    default = Tool(name="default", description="", handler=lambda: "ok")
    router = ToolRouter(registry=registry, default_tool=default)

    assert router.route("missing") == default


def test_tool_router_errors_without_default() -> None:
    registry = ToolRegistry()
    router = ToolRouter(registry=registry)

    with pytest.raises(KeyError):
        router.route("missing")


def test_react_agent_invokes_tool_and_tracks_history() -> None:
    registry = ToolRegistry()
    tool = Tool(name="double", description="", handler=lambda value: value * 2)
    registry.register(tool)
    agent = ReActAgent(router=ToolRouter(registry=registry))

    result = agent.act("double", 3)

    assert result == 6
    assert agent.history == [{"tool": "double", "args": (3,), "kwargs": {}}]
