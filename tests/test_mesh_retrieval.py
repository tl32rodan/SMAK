from __future__ import annotations

import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import pytest

from smak.agent.react import build_llamaindex_react_agent
from smak.agent.tools import MeshRetrievalTool


@dataclass
class FakeIndex:
    data: dict[str, dict[str, Any]]
    lookups: list[str] = field(default_factory=list)

    def search(self, query: str) -> list[dict[str, Any]]:
        return [self.data["code::login"]]

    def get_by_id(self, uid: str) -> dict[str, Any] | None:
        self.lookups.append(uid)
        return self.data.get(uid)


class FakeRegistry:
    def __init__(self, indices: dict[str, FakeIndex]) -> None:
        self.indices = indices

    def get_index(self, name: str) -> FakeIndex:
        return self.indices[name]


def _install_fake_llamaindex(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    fake_tools = ModuleType("llama_index.core.tools")

    class FakeFunctionTool:
        def __init__(self, fn: Any, name: str, description: str) -> None:
            self.fn = fn
            self.name = name
            self.description = description

        @classmethod
        def from_defaults(cls, fn: Any, name: str, description: str) -> "FakeFunctionTool":
            return cls(fn=fn, name=name, description=description)

    class FakeQueryEngineTool:
        def __init__(self, query_engine: Any, name: str, description: str) -> None:
            self.query_engine = query_engine
            self.name = name
            self.description = description

        @classmethod
        def from_defaults(
            cls, query_engine: Any, name: str, description: str
        ) -> "FakeQueryEngineTool":
            return cls(query_engine=query_engine, name=name, description=description)

    fake_tools.FunctionTool = FakeFunctionTool
    fake_tools.QueryEngineTool = FakeQueryEngineTool

    fake_agent_module = ModuleType("llama_index.core.agent")

    class FakeAgent:
        def __init__(self, tools: list[Any], llm: Any | None = None, **kwargs: Any) -> None:
            self.tools = tools
            self.llm = llm
            self.kwargs = kwargs

        @classmethod
        def from_tools(cls, tools: list[Any], llm: Any | None = None, **kwargs: Any) -> "FakeAgent":
            return cls(tools=tools, llm=llm, **kwargs)

    fake_agent_module.ReActAgent = FakeAgent

    fake_core = ModuleType("llama_index.core")
    fake_core.tools = fake_tools
    fake_core.agent = fake_agent_module

    fake_root = ModuleType("llama_index")
    fake_root.core = fake_core

    monkeypatch.setitem(sys.modules, "llama_index", fake_root)
    monkeypatch.setitem(sys.modules, "llama_index.core", fake_core)
    monkeypatch.setitem(sys.modules, "llama_index.core.tools", fake_tools)
    monkeypatch.setitem(sys.modules, "llama_index.core.agent", fake_agent_module)

    return {
        "FakeFunctionTool": FakeFunctionTool,
        "FakeQueryEngineTool": FakeQueryEngineTool,
        "FakeAgent": FakeAgent,
    }


def test_mesh_retrieval_tool_lookup_and_context() -> None:
    code_index = FakeIndex(
        data={
            "code::login": {
                "uid": "code::login",
                "content": "login handler",
                "metadata": {"mesh_relations": ["issue::bug_report_502.md"]},
            }
        }
    )
    issue_index = FakeIndex(
        data={
            "issue::bug_report_502.md": {
                "uid": "issue::bug_report_502.md",
                "content": "Bug report details",
                "metadata": {},
            }
        }
    )
    registry = FakeRegistry({"code": code_index, "issue": issue_index})
    tool = MeshRetrievalTool(registry=registry, index_name="code")

    payload = tool.retrieve("login")

    assert issue_index.lookups == ["issue::bug_report_502.md"]
    assert payload["context"] == [
        {
            "id": "code::login",
            "content": "login handler",
            "metadata": {"mesh_relations": ["issue::bug_report_502.md"]},
            "relation_type": "primary",
            "relation_id": None,
        },
        {
            "id": "issue::bug_report_502.md",
            "content": "Bug report details",
            "metadata": {},
            "relation_type": "related",
            "relation_id": "issue::bug_report_502.md",
        },
    ]


def test_mesh_retrieval_tool_builds_llamaindex_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_llamaindex(monkeypatch)
    registry = FakeRegistry({"code": FakeIndex(data={"code::login": {"uid": "code::login"}})})
    tool = MeshRetrievalTool(registry=registry, index_name="code")

    function_tool = tool.as_function_tool()
    query_tool = tool.as_query_engine_tool()

    assert isinstance(function_tool, fake["FakeFunctionTool"])
    assert function_tool.fn == tool.retrieve
    assert isinstance(query_tool, fake["FakeQueryEngineTool"])
    assert query_tool.query_engine.query("login")["query"] == "login"


def test_build_llamaindex_react_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_llamaindex(monkeypatch)
    registry = FakeRegistry({"code": FakeIndex(data={"code::login": {"uid": "code::login"}})})
    tool = MeshRetrievalTool(registry=registry, index_name="code")

    agent = build_llamaindex_react_agent(
        tool,
        tools=["extra"],
        llm="fake-llm",
        use_query_engine_tool=True,
        verbose=True,
    )
    agent_with_function = build_llamaindex_react_agent(tool, tools=[], llm=None)

    assert isinstance(agent, fake["FakeAgent"])
    assert agent.tools[0] == "extra"
    assert agent.tools[1].name == tool.name
    assert agent.llm == "fake-llm"
    assert agent.kwargs["verbose"] is True
    assert isinstance(agent_with_function, fake["FakeAgent"])
    assert isinstance(agent_with_function.tools[0], fake["FakeFunctionTool"])
    assert agent_with_function.tools[0].name == tool.name
