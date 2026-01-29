"""ReAct-style agent stub."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smak.agent.router import ToolRouter


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
