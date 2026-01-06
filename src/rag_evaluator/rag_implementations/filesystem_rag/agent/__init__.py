"""Filesystem RAG agent components."""

from rag_evaluator.rag_implementations.filesystem_rag.agent.agent import (
    AgentResponse,
    FilesystemRAGAgent,
    ReasoningStep,
)
from rag_evaluator.rag_implementations.filesystem_rag.agent.cache import (
    SessionCache,
)
from rag_evaluator.rag_implementations.filesystem_rag.agent.prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_COMPACT,
    TOOL_EXAMPLES,
    format_final_answer_prompt,
    format_limit_reached_prompt,
    format_reasoning_step,
    format_system_prompt,
    format_tool_result,
    format_user_message,
)
from rag_evaluator.rag_implementations.filesystem_rag.agent.router import (
    QueryRouter,
    RoutingResult,
    SearchMode,
)
from rag_evaluator.rag_implementations.filesystem_rag.agent.tools import (
    FilesystemRAGTools,
)

__all__ = [
    # agent
    "FilesystemRAGAgent",
    "AgentResponse",
    "ReasoningStep",
    # tools
    "FilesystemRAGTools",
    # cache
    "SessionCache",
    # router
    "QueryRouter",
    "RoutingResult",
    "SearchMode",
    # prompts
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_COMPACT",
    "TOOL_EXAMPLES",
    "format_system_prompt",
    "format_user_message",
    "format_tool_result",
    "format_final_answer_prompt",
    "format_limit_reached_prompt",
    "format_reasoning_step",
]
