from __future__ import annotations

import sys
import unittest
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Generator
from unittest.mock import patch

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


@contextmanager
def install_fake_llamaindex() -> Generator[dict[str, Any], None, None]:
    fake_tools = ModuleType("llama_index.core.tools")
    fake_query_engine = ModuleType("llama_index.core.query_engine")

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

    class FakeBaseQueryEngine:
        def query(self, query: str) -> Any:
            return self._query(query)

    fake_tools.FunctionTool = FakeFunctionTool
    fake_tools.QueryEngineTool = FakeQueryEngineTool
    fake_query_engine.BaseQueryEngine = FakeBaseQueryEngine

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
    fake_core.query_engine = fake_query_engine
    fake_core.agent = fake_agent_module

    fake_root = ModuleType("llama_index")
    fake_root.core = fake_core

    with patch.dict(
        sys.modules,
        {
            "llama_index": fake_root,
            "llama_index.core": fake_core,
            "llama_index.core.tools": fake_tools,
            "llama_index.core.query_engine": fake_query_engine,
            "llama_index.core.agent": fake_agent_module,
        },
    ):
        yield {
            "FakeFunctionTool": FakeFunctionTool,
            "FakeQueryEngineTool": FakeQueryEngineTool,
            "FakeBaseQueryEngine": FakeBaseQueryEngine,
            "FakeAgent": FakeAgent,
        }


class TestMeshRetrievalTool(unittest.TestCase):
    def test_mesh_retrieval_tool_lookup_and_context(self) -> None:
        code_index = FakeIndex(
            data={
                "code::login": {
                    "uid": "code::login",
                    "content": "login handler",
                    "metadata": {"mesh_relations": ["issue:bug_report_502.md"]},
                }
            }
        )
        issue_index = FakeIndex(
            data={
                "issue:bug_report_502.md": {
                    "uid": "issue:bug_report_502.md",
                    "content": "Bug report details",
                    "metadata": {},
                }
            }
        )
        registry = FakeRegistry({"code": code_index, "issue": issue_index})
        tool = MeshRetrievalTool(registry=registry, index_name="code")

        payload = tool.retrieve("login")

        self.assertEqual(issue_index.lookups, ["issue:bug_report_502.md"])
        self.assertEqual(
            payload["context"],
            [
                {
                    "id": "code::login",
                    "content": "login handler",
                    "metadata": {"mesh_relations": ["issue:bug_report_502.md"]},
                    "relation_type": "primary",
                    "relation_id": None,
                },
                {
                    "id": "issue:bug_report_502.md",
                    "content": "Bug report details",
                    "metadata": {},
                    "relation_type": "related",
                    "relation_id": "issue:bug_report_502.md",
                },
            ],
        )

    def test_mesh_retrieval_tool_builds_llamaindex_tools(self) -> None:
        with install_fake_llamaindex() as fake:
            registry = FakeRegistry(
                {"code": FakeIndex(data={"code::login": {"uid": "code::login"}})}
            )
            tool = MeshRetrievalTool(registry=registry, index_name="code")

            function_tool = tool.as_function_tool()
            query_tool = tool.as_query_engine_tool()

            self.assertIsInstance(function_tool, fake["FakeFunctionTool"])
            self.assertEqual(function_tool.fn, tool.retrieve)
            self.assertIsInstance(query_tool, fake["FakeQueryEngineTool"])
            self.assertIsInstance(query_tool.query_engine, fake["FakeBaseQueryEngine"])
            self.assertEqual(query_tool.query_engine.query("login")["query"], "login")

    def test_build_llamaindex_react_agent(self) -> None:
        with install_fake_llamaindex() as fake:
            registry = FakeRegistry(
                {"code": FakeIndex(data={"code::login": {"uid": "code::login"}})}
            )
            tool = MeshRetrievalTool(registry=registry, index_name="code")

            agent = build_llamaindex_react_agent(
                tool,
                tools=["extra"],
                llm="fake-llm",
                use_query_engine_tool=True,
                verbose=True,
            )
            agent_with_function = build_llamaindex_react_agent(tool, tools=[], llm=None)

            self.assertIsInstance(agent, fake["FakeAgent"])
            self.assertEqual(agent.tools[0], "extra")
            self.assertEqual(agent.tools[1].name, tool.name)
            self.assertEqual(agent.llm, "fake-llm")
            self.assertIs(agent.kwargs["verbose"], True)
            self.assertIsInstance(agent_with_function, fake["FakeAgent"])
            self.assertIsInstance(agent_with_function.tools[0], fake["FakeFunctionTool"])
            self.assertEqual(agent_with_function.tools[0].name, tool.name)


if __name__ == "__main__":
    unittest.main()
