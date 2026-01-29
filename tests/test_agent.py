from dataclasses import dataclass

import pytest

from smak.agent.react import ReActAgent
from smak.agent.router import IndexRouter, ToolRouter
from smak.agent.tools import MeshSearchTool, Tool, ToolRegistry


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


def test_index_router_routes_with_classifier() -> None:
    router = IndexRouter(
        index_names=["code", "issue"],
        classifier=lambda query, names: [name for name in names if name in query],
    )

    assert router.route("code search") == ["code"]


def test_mesh_search_tool_expands_relations() -> None:
    @dataclass
    class FakeIndex:
        data: dict[str, dict]

        def search(self, query: str) -> list[dict]:
            return [self.data["code::login"]]

        def get_by_id(self, uid: str) -> dict | None:
            return self.data.get(uid)

    class FakeRegistry:
        def __init__(self) -> None:
            self.indices = {
                "code": FakeIndex(
                    data={
                        "code::login": {
                            "payload": {"relations": ["issue::101"]},
                            "uid": "code::login",
                        },
                    }
                ),
                "issue": FakeIndex(
                    data={
                        "issue::101": {"payload": {"content": "issue body"}, "uid": "issue::101"}
                    }
                ),
            }

        def get_index(self, name: str) -> FakeIndex:
            return self.indices[name]

    tool = MeshSearchTool(registry=FakeRegistry())

    results = tool.search("code", "login")

    assert [result["uid"] for result in results] == ["code::login", "issue::101"]
