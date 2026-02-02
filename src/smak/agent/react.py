"""ReAct-style agent stub."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from smak.agent.router import ToolRouter
from smak.agent.tools import MeshRetrievalTool
from smak.utils.llm_parser import ensure_action_payload, parse_json_from_text

logger = logging.getLogger(__name__)


@dataclass
class ReActAgent:
    """Agent that routes to tools and tracks actions."""

    router: ToolRouter
    history: list[dict[str, Any]] = field(default_factory=list)

    def act(self, tool_name: str, *args: Any, **kwargs: Any) -> Any:
        tool = self.router.route(tool_name)
        result = tool(*args, **kwargs)
        self.history.append({"tool": tool.name, "args": args, "kwargs": kwargs})
        return result

    def step(
        self,
        prompt: str,
        llm: Callable[[str], str],
        *,
        retries: int = 2,
    ) -> Any:
        """Run a single ReAct step by querying an LLM and invoking a tool."""

        last_error: Exception | None = None
        current_prompt = prompt
        for attempt in range(retries + 1):
            response = llm(current_prompt)
            try:
                payload = parse_json_from_text(response)
                action = ensure_action_payload(payload)
                return self.act(action["tool"], *action["args"], **action["kwargs"])
            except Exception as exc:  # pragma: no cover - defensive for unexpected failures
                last_error = exc
                logger.warning(
                    "Failed to parse LLM response on attempt %s: %s", attempt + 1, exc
                )
                current_prompt = (
                    f"{prompt}\n\nThe previous response could not be parsed: {exc}. "
                    "Please respond with valid JSON only."
                )
        raise ValueError("Unable to parse LLM response after retries.") from last_error


def build_llamaindex_react_agent(
    mesh_tool: MeshRetrievalTool,
    *,
    tools: Sequence[Any] | None = None,
    llm: Any | None = None,
    use_query_engine_tool: bool = False,
    **kwargs: Any,
) -> Any:
    agent_cls = _require_llamaindex_agent()
    llama_tools = list(tools or [])
    if use_query_engine_tool:
        llama_tools.append(mesh_tool.as_query_engine_tool())
    else:
        llama_tools.append(mesh_tool.as_function_tool())
    if hasattr(agent_cls, "from_tools"):
        return agent_cls.from_tools(llama_tools, llm=llm, **kwargs)
    return agent_cls(llama_tools, llm=llm, **kwargs)


def _require_llamaindex_agent() -> Any:
    try:
        return __import__("llama_index.core.agent", fromlist=["ReActAgent"]).__dict__[
            "ReActAgent"
        ]
    except (ModuleNotFoundError, KeyError, AttributeError) as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "LlamaIndex is required to build the ReActAgent. "
            "Install 'llama-index-core' to use this feature."
        ) from exc
