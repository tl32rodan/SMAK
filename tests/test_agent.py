import unittest
from dataclasses import dataclass

from smak.agent.react import ReActAgent
from smak.agent.router import IndexRouter, ToolRouter
from smak.agent.tools import MeshSearchTool, Tool, ToolRegistry
from smak.utils.llm_parser import parse_json_from_text


class TestToolRegistry(unittest.TestCase):
    def test_tool_registry_registers_and_lists(self) -> None:
        registry = ToolRegistry()
        tool = Tool(name="echo", description="", handler=lambda value: value)

        registry.register(tool)

        self.assertEqual(registry.get("echo"), tool)
        self.assertEqual(registry.list_tools(), [tool])


class TestToolRouter(unittest.TestCase):
    def test_tool_router_uses_default_tool(self) -> None:
        registry = ToolRegistry()
        default = Tool(name="default", description="", handler=lambda: "ok")
        router = ToolRouter(registry=registry, default_tool=default)

        self.assertEqual(router.route("missing"), default)

    def test_tool_router_errors_without_default(self) -> None:
        registry = ToolRegistry()
        router = ToolRouter(registry=registry)

        with self.assertRaises(KeyError):
            router.route("missing")


class TestReActAgent(unittest.TestCase):
    def test_react_agent_invokes_tool_and_tracks_history(self) -> None:
        registry = ToolRegistry()
        tool = Tool(name="double", description="", handler=lambda value: value * 2)
        registry.register(tool)
        agent = ReActAgent(router=ToolRouter(registry=registry))

        result = agent.act("double", 3)

        self.assertEqual(result, 6)
        self.assertEqual(agent.history, [{"tool": "double", "args": (3,), "kwargs": {}}])

    def test_react_agent_step_retries_with_json_parse(self) -> None:
        registry = ToolRegistry()
        tool = Tool(name="double", description="", handler=lambda value: value * 2)
        registry.register(tool)
        agent = ReActAgent(router=ToolRouter(registry=registry))

        responses = iter(
            [
                "Here is the JSON you asked for: {\"tool\": \"double\", \"args\": [4],}",
                "{\"tool\": \"double\", \"args\": [4]}",
            ]
        )

        def fake_llm(prompt: str) -> str:
            return next(responses)

        result = agent.step("prompt", fake_llm, retries=1)

        self.assertEqual(result, 8)
        self.assertEqual(agent.history[-1]["tool"], "double")


class TestIndexRouter(unittest.TestCase):
    def test_index_router_routes_with_classifier(self) -> None:
        router = IndexRouter(
            index_names=["code", "issue"],
            classifier=lambda query, names: [name for name in names if name in query],
        )

        self.assertEqual(router.route("code search"), ["code"])


class TestMeshSearchTool(unittest.TestCase):
    def test_mesh_search_tool_expands_relations(self) -> None:
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
                                "payload": {"relations": ["issue:101"]},
                                "uid": "code::login",
                            },
                        }
                    ),
                    "issue": FakeIndex(
                        data={
                            "issue:101": {
                                "payload": {"content": "issue body"},
                                "uid": "issue:101",
                            }
                        }
                    ),
                }

            def get_index(self, name: str) -> FakeIndex:
                return self.indices[name]

        tool = MeshSearchTool(registry=FakeRegistry())

        results = tool.search("code", "login")

        self.assertEqual([result["uid"] for result in results], ["code::login", "issue:101"])


class TestLLMParser(unittest.TestCase):
    def test_parse_json_from_text_handles_filler(self) -> None:
        parsed = parse_json_from_text("Sure! {\"tool\": \"noop\"} Thanks.")

        self.assertEqual(parsed, {"tool": "noop"})


if __name__ == "__main__":
    unittest.main()
