"""What the pipeline needs, in one object.

Constructed once per process (the expensive clients) with the per-request pieces —
repositories bound to a transaction — passed in at call time. A single bundle rather than
eleven constructor arguments: adding a stage should not mean editing every call site between
the container and the node that needs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aegis.domain.policies.budget import BudgetPolicy
from aegis.domain.policies.confidence import ConfidenceGate
from aegis.rag.retrieval.compression import CompressionConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from aegis.rag.guardrails import Guardrails
    from aegis.rag.llm.router import LLMRouter
    from aegis.rag.retrieval.hybrid import HybridRetriever
    from aegis.rag.retrieval.query import QueryPlanner


@dataclass(slots=True)
class PipelineDeps:
    llm: LLMRouter
    planner: QueryPlanner
    retriever: HybridRetriever
    guardrails: Guardrails
    gate: ConfidenceGate = field(default_factory=ConfidenceGate)
    budget: BudgetPolicy = field(default_factory=BudgetPolicy)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    assistant_name: str = "Aegis"
    max_output_tokens: int = 2000
    temperature: float = 0.1

    @property
    def count_tokens(self) -> Callable[[str], int]:
        """Bound token counter for the budgeting code, which should not need the router."""
        return lambda text: self.llm.count_tokens(text)
