"""Dependency objects injected into Pydantic AI agent runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import httpx


ProgressCallback = Callable[[str], None]


@dataclass(slots=True)
class StockAgentDeps:
    """Per-run dependencies and mutable scratchpad state for the agent."""

    http_client: httpx.AsyncClient | None = None
    progress_callback: ProgressCallback | None = None
    reasoning_chain: list[str] = field(default_factory=list)
    tool_calls: int = 0
    max_tool_calls: int = 6

    def emit_progress(self, message: str) -> None:
        """Send a short progress message to the CLI, if one is registered."""

        if self.progress_callback is not None:
            self.progress_callback(message)

    def record_reasoning(self, reasoning: str, next_action: str) -> str:
        """Persist a reasoning step for inclusion in the final report."""

        entry = f"{reasoning.strip()} Next action: {next_action.strip()}"
        self.reasoning_chain.append(entry)
        return entry

    def count_tool_call(self) -> None:
        """Track tool usage so the ReAct loop can stay within its budget."""

        self.tool_calls += 1

    def has_tool_budget(self) -> bool:
        """Return whether another tool call is allowed for this run."""

        return self.tool_calls < self.max_tool_calls
