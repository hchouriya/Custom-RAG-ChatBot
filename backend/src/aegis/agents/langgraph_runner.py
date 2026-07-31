"""Optional LangGraph runner over the same nodes.

The pre-generation stages form a small graph: guard → plan → (retrieve | skip) → gate →
(prompt | refuse). ``pipeline.prepare`` runs that graph directly in about fifteen lines, which
is the default because it adds no dependency, is trivially debuggable in a stack trace, and
streams without an adapter.

LangGraph earns its place when the graph stops being a line — checkpointed multi-step
research loops, human-in-the-loop interrupts, sub-agents that call each other. This module
exists so that switching is a configuration change rather than a rewrite: it builds a
``StateGraph`` from the *identical* node functions, so both runners share one implementation
of every stage and cannot drift.

Enabled with ``ORCHESTRATOR=langgraph`` and the ``langgraph`` extra installed. Absent either,
the native runner is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aegis.agents.nodes import build_prompt, gate, guard_input, plan_query, retrieve
from aegis.core.errors import ConfigurationError
from aegis.core.logging import get_logger
from aegis.core.telemetry import current_trace_id
from aegis.domain.enums import AnswerStatus

if TYPE_CHECKING:
    from aegis.agents.deps import PipelineDeps
    from aegis.agents.state import AnswerRequest, AnswerState

logger = get_logger(__name__)


def is_available() -> bool:
    try:
        import langgraph  # noqa: F401
    except ImportError:
        return False
    return True


class LangGraphPreparer:
    """Runs the pre-generation stages as a compiled LangGraph.

    Generation stays outside the graph. LangGraph can stream, but wrapping a token stream in
    a graph's own streaming protocol buys nothing here and costs the direct back-pressure
    path from provider to client that time-to-first-token depends on.
    """

    def __init__(self, deps: PipelineDeps) -> None:
        if not is_available():  # pragma: no cover - depends on the environment
            raise ConfigurationError(
                "ORCHESTRATOR=langgraph requires the 'langgraph' extra: "
                "pip install 'aegis[langgraph]'"
            )
        from langgraph.graph import END, StateGraph

        self._deps = deps

        # The state is a single mutable object, so every node returns an empty update dict:
        # LangGraph's reducer semantics are for accumulating partial state, which is not the
        # shape of a pipeline whose stages each own one field.
        def wrap(fn: Any) -> Any:
            async def node(state: dict[str, Any]) -> dict[str, Any]:
                await fn(state["state"], self._deps)
                return {}

            node.__name__ = fn.__name__
            return node

        graph: Any = StateGraph(dict)
        graph.add_node("guard", wrap(guard_input))
        graph.add_node("plan", wrap(plan_query))
        graph.add_node("retrieve", wrap(retrieve))
        graph.add_node("gate", wrap(gate))
        graph.add_node("prompt", wrap(build_prompt))

        graph.set_entry_point("guard")
        graph.add_edge("guard", "plan")
        graph.add_conditional_edges(
            "plan",
            lambda s: "retrieve" if s["state"].intent.needs_retrieval else "prompt",
            {"retrieve": "retrieve", "prompt": "prompt"},
        )
        graph.add_edge("retrieve", "gate")
        graph.add_conditional_edges(
            "gate",
            # A refusal still builds a prompt only when it is a clarification; a flat refusal
            # has no prompt to build and ends here.
            lambda s: END if s["state"].status is AnswerStatus.NO_ANSWER else "prompt",
            {END: END, "prompt": "prompt"},
        )
        graph.add_edge("prompt", END)
        self._graph = graph.compile()

    async def prepare(self, request: AnswerRequest) -> AnswerState:
        from aegis.agents.state import AnswerState

        state = AnswerState(request=request, trace_id=current_trace_id())
        await self._graph.ainvoke({"state": state})
        return state
