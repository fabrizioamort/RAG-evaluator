"""Filesystem RAG agent implementation.

This module implements the ReAct (Reasoning + Acting) agent loop
for navigating the prepared filesystem and answering questions.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from rag_evaluator.common.llm_utils import get_safe_llm_params, is_transient_llm_error
from rag_evaluator.config import settings
from rag_evaluator.rag_implementations.filesystem_rag.agent.cache import SessionCache
from rag_evaluator.rag_implementations.filesystem_rag.agent.prefetch import (
    build_prefetch_context,
)
from rag_evaluator.rag_implementations.filesystem_rag.agent.prompts import (
    TOOL_MARKUP_RETRY_PROMPT,
    format_answer_retry_prompt,
    format_evidence_nudge_prompt,
    format_limit_reached_prompt,
    format_system_prompt,
    format_tool_result,
    format_user_message,
)
from rag_evaluator.rag_implementations.filesystem_rag.agent.router import (
    QueryRouter,
    SearchMode,
)
from rag_evaluator.rag_implementations.filesystem_rag.agent.tools import (
    FilesystemRAGTools,
)


@dataclass
class AgentResponse:
    """Response from the filesystem RAG agent.

    Attributes:
        answer: The generated answer to the question
        context: List of context chunks used to generate the answer
        metadata: Additional metadata about the query execution
    """

    answer: str
    context: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningStep:
    """A single step in the agent's reasoning trace.

    Attributes:
        iteration: The iteration number
        tool_name: Name of the tool called (if any)
        tool_args: Arguments passed to the tool
        tool_result: Result from the tool (truncated)
        thought: Agent's reasoning (if extractable)
    """

    iteration: int
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: str | None = None
    thought: str | None = None


_LLM_MAX_ATTEMPTS = 3
_LLM_RETRY_BASE_DELAY_SECONDS = 1.0
_CONTEXT_CHUNK_MAX_CHARS = 20_000
_NAVIGATION_CONTEXT_PREFIXES = ("_index/questions/", "_index/passages/")
_REFUSAL_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^\s*i(?:'m| am) sorry\b",
        r"^\s*i can(?:not|'t) (?:help|assist|answer|provide|comply)\b",
        r"^\s*i (?:will not|won't) (?:help|assist|answer|provide)\b",
        r"^\s*(?:sorry,? )?(?:but )?(?:i )?can(?:not|'t) assist\b",
    )
)
# Raw tool-call wire formats leaked as plain content (observed: DeepSeek DSML
# markup with fullwidth vertical bars U+FF5C). The tool-name whitelist keeps
# the invoke check conservative so answers that merely mention a tool name are
# not flagged.
_TOOL_MARKUP_RE = re.compile(
    r"(DSML|<\|?｜?tool_calls?|</?tool_call>|<\|?｜?invoke\b"
    r'|\binvoke name="(read_file|grep_search|search_passages'
    r'|list_directory|find_files|get_file_info)")',
    re.IGNORECASE,
)


def unusable_answer_reason(answer: str) -> str | None:
    """Classify final answers that warrant a single corrective retry.

    Returns "empty", "non_english", "tool_markup", "refusal", or None for
    usable answers. Detection is deliberately conservative: only the failure
    shapes observed on the legal benchmark (empty answers, CJK refusals,
    leaked tool-call markup, refusal openings).
    """
    stripped = answer.strip()
    if not stripped:
        return "empty"

    cjk_count = sum(1 for ch in stripped if chr(0x4E00) <= ch <= chr(0x9FFF))
    if cjk_count > 0.2 * len(stripped):
        return "non_english"

    if _TOOL_MARKUP_RE.search(stripped):
        return "tool_markup"

    if any(pattern.search(stripped) for pattern in _REFUSAL_PATTERNS):
        return "refusal"

    return None


class FilesystemRAGAgent:
    """LLM-guided filesystem navigation agent.

    Uses a ReAct (Reasoning + Acting) loop to:
    1. Analyze the query and determine search strategy
    2. Navigate the prepared filesystem using tools
    3. Gather relevant information
    4. Synthesize an answer

    Usage:
        agent = FilesystemRAGAgent(
            prepared_path="/path/to/prepared",
            llm_model="gpt-4o-mini"
        )
        response = agent.query("What are the main challenges with RAG?")
        print(response.answer)
    """

    def __init__(
        self,
        prepared_path: str,
        llm_model: str = "gpt-4o-mini",
        max_iterations: int = 10,
        max_tool_calls: int = 20,
        max_file_reads: int = 10,
        client: OpenAI | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        """Initialize the filesystem RAG agent.

        Args:
            prepared_path: Path to the prepared filesystem root
            llm_model: OpenAI model to use for reasoning
            max_iterations: Maximum ReAct loop iterations
            max_tool_calls: Maximum total tool calls per query
            max_file_reads: Maximum file read operations per query
            client: Optional pre-configured OpenAI client
            reasoning_effort: Reasoning effort level (low/medium/high)
        """
        self.prepared_path = prepared_path
        self.llm_model = llm_model
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.max_file_reads = max_file_reads
        self.reasoning_effort = reasoning_effort

        # Initialize components
        self.tools = FilesystemRAGTools(prepared_path)
        self.cache = SessionCache(prepared_path)
        self.router = QueryRouter()

        # Initialize or use provided client
        if client is not None:
            self.client = client
        else:
            self.client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                timeout=settings.openai_timeout,
            )

        # Warm the cache
        self.cache.warm()

        self._llm_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def reset_llm_usage(self) -> None:
        """Reset accumulated usage for the next logical operation."""
        self._llm_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def get_llm_usage(self) -> dict[str, int]:
        """Return accumulated chat completion usage."""
        return dict(self._llm_usage)

    def _record_llm_usage(self, response: Any) -> None:
        """Accumulate provider-reported token usage from a chat response."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return

        prompt_tokens = self._usage_int(usage, "prompt_tokens", "input_tokens")
        completion_tokens = self._usage_int(usage, "completion_tokens", "output_tokens")
        total_tokens = self._usage_int(usage, "total_tokens")
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        self._llm_usage["prompt_tokens"] += prompt_tokens
        self._llm_usage["completion_tokens"] += completion_tokens
        self._llm_usage["total_tokens"] += total_tokens

    def _usage_int(self, usage: Any, *keys: str) -> int:
        """Read an integer usage field from dict-like or object-like usage."""
        for key in keys:
            if isinstance(usage, dict):
                value = usage.get(key)
            else:
                value = getattr(usage, key, None)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return 0

    def _build_prefetch_context(self, question: str, max_candidates: int = 5) -> dict[str, Any]:
        """Build deterministic BM25 candidate context before LLM navigation starts."""
        return build_prefetch_context(self.tools, question, max_candidates=max_candidates)

    def _context_chunk_from_tool_result(
        self,
        file_path: str,
        result: Any,
    ) -> str | None:
        """Return a bounded evidence chunk for evaluation metadata."""
        normalized_path = file_path.replace("\\", "/").lstrip("./")
        if normalized_path.startswith(_NAVIGATION_CONTEXT_PREFIXES):
            return None

        if isinstance(result, dict) and "content" in result:
            content = str(result["content"])
        elif isinstance(result, str):
            content = result
        else:
            return None

        if len(content) <= _CONTEXT_CHUNK_MAX_CHARS:
            return content

        return (
            content[:_CONTEXT_CHUNK_MAX_CHARS].rstrip()
            + "\n... [retrieved context chunk truncated]"
        )

    def query(self, question: str) -> AgentResponse:
        """Answer a question by navigating the filesystem.

        Args:
            question: The question to answer

        Returns:
            AgentResponse containing answer, context, and metadata
        """
        start_time = time.time()
        self.reset_llm_usage()

        # Route query to determine search mode
        routing_result = self.router.route(question)
        search_mode = routing_result.mode
        strategy_hint = routing_result.strategy_hint

        # Build system prompt
        prefetch = self._build_prefetch_context(question)
        initial_context = self.cache.get_initial_context()
        if prefetch["chunks"]:
            candidate_context = "\n\n".join(prefetch["chunks"])
            initial_context = (
                f"{initial_context}\n\n"
                "=== _index/passages/bm25_candidates ===\n"
                "These candidate snippets were selected by BM25 passage search. Use "
                "them when relevant, and verify with filesystem tools if the answer "
                "is uncertain.\n\n"
                f"{candidate_context}"
            )
        system_prompt = format_system_prompt(
            strategy_hint=strategy_hint,
            initial_context=initial_context,
            max_tool_calls=self.max_tool_calls,
            max_file_reads=self.max_file_reads,
        )

        # Initialize conversation
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": format_user_message(question)},
        ]

        # Tracking
        reasoning_trace: list[ReasoningStep] = []
        files_read: list[str] = []
        context_chunks: list[str] = list(prefetch["chunks"])
        context_sources: list[str] = list(prefetch["sources"])
        tool_call_count = 0
        file_read_count = 0
        markup_retry_used = False
        evidence_nudge_used = False

        # ReAct loop
        for iteration in range(self.max_iterations):
            # Call LLM
            response = self._call_llm(messages)

            # Check if LLM wants to use tools
            if response.choices[0].message.tool_calls:
                tool_calls = response.choices[0].message.tool_calls

                # Add assistant message with tool calls
                messages.append(response.choices[0].message.model_dump())

                # Process each tool call
                for tool_call_index, tool_call in enumerate(tool_calls):
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    # Check tool call limit
                    if tool_call_count >= self.max_tool_calls:
                        self._append_limit_tool_messages(
                            messages,
                            tool_calls[tool_call_index:],
                            "tool_calls",
                        )
                        return self._synthesize_partial_answer(
                            messages=messages,
                            context_chunks=context_chunks,
                            reasoning_trace=reasoning_trace,
                            search_mode=search_mode,
                            files_read=files_read,
                            context_sources=context_sources,
                            tool_call_count=tool_call_count,
                            start_time=start_time,
                            prefetch=prefetch,
                            limit_type="tool_calls",
                            iterations=iteration + 1,
                        )

                    # Check file read limit
                    if tool_name == "read_file" and file_read_count >= self.max_file_reads:
                        self._append_limit_tool_messages(
                            messages,
                            tool_calls[tool_call_index:],
                            "file_reads",
                        )
                        return self._synthesize_partial_answer(
                            messages=messages,
                            context_chunks=context_chunks,
                            reasoning_trace=reasoning_trace,
                            search_mode=search_mode,
                            files_read=files_read,
                            context_sources=context_sources,
                            tool_call_count=tool_call_count,
                            start_time=start_time,
                            prefetch=prefetch,
                            limit_type="file_reads",
                            iterations=iteration + 1,
                        )

                    # Execute tool
                    result = self._execute_tool(tool_name, tool_args)
                    tool_call_count += 1

                    # Track file reads
                    if tool_name == "read_file":
                        file_read_count += 1
                        file_path = tool_args.get("path", "")
                        files_read.append(file_path)

                        # Store bounded evidence context only. Navigation indexes
                        # can be useful to the agent, but they are not evidence for
                        # downstream judges.
                        context_chunk = self._context_chunk_from_tool_result(file_path, result)
                        if context_chunk is not None:
                            context_chunks.append(context_chunk)
                            context_sources.append(file_path)

                    # Format result for conversation
                    result_str = format_tool_result(tool_name, result)

                    # Track reasoning
                    reasoning_trace.append(
                        ReasoningStep(
                            iteration=iteration,
                            tool_name=tool_name,
                            tool_args=tool_args,
                            tool_result=result_str[:500] if len(result_str) > 500 else result_str,
                        )
                    )

                    # Add tool result to conversation
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_str,
                        }
                    )

            else:
                # LLM provided final answer (no tool calls)
                answer = response.choices[0].message.content or ""
                retry_reason = unusable_answer_reason(answer)

                # Leaked tool-call markup means the model tried to act, so give
                # it one chance to re-issue the call through the tool mechanism
                # instead of forcing a plain-completion answer.
                if (
                    retry_reason == "tool_markup"
                    and not markup_retry_used
                    and iteration + 1 < self.max_iterations
                    and tool_call_count < self.max_tool_calls
                ):
                    markup_retry_used = True
                    messages.append({"role": "assistant", "content": answer})
                    messages.append({"role": "user", "content": TOOL_MARKUP_RETRY_PROMPT})
                    continue

                # A usable answer built on fewer than two document reads gets a
                # single verification nudge before it is accepted.
                distinct_docs_read = {
                    normalized
                    for normalized in (f.replace("\\", "/").lstrip("./") for f in files_read)
                    if normalized.startswith("documents/")
                }
                if (
                    retry_reason is None
                    and len(distinct_docs_read) < 2
                    and not evidence_nudge_used
                    and iteration + 1 < self.max_iterations
                    and tool_call_count < self.max_tool_calls
                    and file_read_count < self.max_file_reads
                ):
                    evidence_nudge_used = True
                    messages.append({"role": "assistant", "content": answer})
                    messages.append(
                        {
                            "role": "user",
                            "content": format_evidence_nudge_prompt(len(distinct_docs_read)),
                        }
                    )
                    continue

                answer_retries = 0
                if retry_reason is not None:
                    answer = self._retry_unusable_answer(messages, answer, retry_reason)
                    answer_retries = 1

                query_time = time.time() - start_time

                return AgentResponse(
                    answer=answer,
                    context=context_chunks,
                    metadata={
                        "files_read": files_read,
                        "context_sources": context_sources,
                        "answer_retries": answer_retries,
                        "answer_retry_reason": retry_reason,
                        "markup_recovery_used": markup_retry_used,
                        "evidence_nudge_used": evidence_nudge_used,
                        "llm_request_params": self._resolved_request_params(),
                        "tool_calls": tool_call_count,
                        "reasoning_trace": [
                            {
                                "iteration": s.iteration,
                                "tool": s.tool_name,
                                "args": s.tool_args,
                                "result_preview": s.tool_result,
                            }
                            for s in reasoning_trace
                        ],
                        "search_mode": search_mode.value,
                        "iterations": iteration + 1,
                        "query_time": query_time,
                        "routing_confidence": routing_result.confidence,
                        "prefetch_terms": prefetch["terms"],
                        "prefetch_candidates": prefetch["candidates"],
                        "token_usage": self.get_llm_usage(),
                    },
                )

        # Max iterations reached - force answer
        return self._synthesize_partial_answer(
            messages=messages,
            context_chunks=context_chunks,
            reasoning_trace=reasoning_trace,
            search_mode=search_mode,
            files_read=files_read,
            context_sources=context_sources,
            tool_call_count=tool_call_count,
            start_time=start_time,
            prefetch=prefetch,
            limit_type="iterations",
            iterations=self.max_iterations,
        )

    def _append_limit_tool_messages(
        self,
        messages: list[dict[str, Any]],
        tool_calls: list[Any],
        limit_type: str,
    ) -> None:
        """Add tool responses for pending tool calls after a limit is hit."""
        for pending_call in tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": pending_call.id,
                    "content": format_limit_reached_prompt(limit_type),
                }
            )

    def _resolved_request_params(self) -> dict[str, Any]:
        """Snapshot the generation parameters actually sent to the provider.

        temperature None means the parameter was omitted, so the provider
        default applied.
        """
        kwargs = get_safe_llm_params(self.llm_model, reasoning_effort=self.reasoning_effort)
        return {
            "model": self.llm_model,
            "temperature": kwargs.get("temperature"),
            "reasoning_effort": kwargs.get("reasoning_effort"),
        }

    def _call_llm(self, messages: list[dict[str, Any]]) -> Any:
        """Call the LLM with tool definitions.

        Args:
            messages: Conversation messages

        Returns:
            OpenAI API response
        """
        # Get tool definitions
        tool_definitions = self.tools.get_tool_definitions()

        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": self.llm_model,
            "messages": messages,
            "tools": tool_definitions,
            "tool_choice": "auto",
        }

        kwargs = get_safe_llm_params(
            self.llm_model, reasoning_effort=self.reasoning_effort, **kwargs
        )

        return self._create_chat_completion_with_retries(kwargs)

    def _retry_unusable_answer(
        self,
        messages: list[dict[str, Any]],
        answer: str,
        reason: str,
    ) -> str:
        """Request one corrected final answer after an unusable one.

        Reuses the gathered conversation so the retry is a single plain
        completion without further tool use.
        """
        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": format_answer_retry_prompt(reason)})

        kwargs: dict[str, Any] = {"model": self.llm_model, "messages": messages}
        kwargs = get_safe_llm_params(
            self.llm_model, temperature=0.0, reasoning_effort=self.reasoning_effort, **kwargs
        )
        response = self._create_chat_completion_with_retries(kwargs)
        return response.choices[0].message.content or ""

    def _create_chat_completion_with_retries(self, kwargs: dict[str, Any]) -> Any:
        """Call the chat completion API, retrying transient provider failures."""
        for attempt in range(_LLM_MAX_ATTEMPTS):
            try:
                response = self.client.chat.completions.create(**kwargs)
                self._record_llm_usage(response)
                return response
            except Exception as exc:
                if attempt < _LLM_MAX_ATTEMPTS - 1 and is_transient_llm_error(exc):
                    delay = _LLM_RETRY_BASE_DELAY_SECONDS * (2**attempt)
                    if delay > 0:
                        time.sleep(delay)
                    continue
                raise

        raise RuntimeError("LLM completion failed without raising an exception")

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Execute a tool and return the result.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments for the tool

        Returns:
            Tool execution result
        """
        try:
            if tool_name == "list_directory":
                return self.tools.list_directory(**args)
            elif tool_name == "read_file":
                return self.tools.read_file(**args)
            elif tool_name == "grep_search":
                return self.tools.grep_search(**args)
            elif tool_name == "search_passages":
                return self.tools.search_passages(**args)
            elif tool_name == "find_files":
                return self.tools.find_files(**args)
            elif tool_name == "get_file_info":
                return self.tools.get_file_info(**args)
            else:
                return f"Error: Unknown tool '{tool_name}'"
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"

    def _synthesize_partial_answer(
        self,
        messages: list[dict[str, Any]],
        context_chunks: list[str],
        reasoning_trace: list[ReasoningStep],
        search_mode: SearchMode,
        files_read: list[str],
        context_sources: list[str],
        tool_call_count: int,
        start_time: float,
        prefetch: dict[str, Any],
        limit_type: str,
        iterations: int,
    ) -> AgentResponse:
        """Synthesize an answer when max iterations reached.

        Args:
            messages: Conversation messages
            context_chunks: Collected context
            reasoning_trace: Reasoning steps
            search_mode: Query search mode
            files_read: List of files read
            tool_call_count: Number of tool calls made
            start_time: Query start time

        Returns:
            AgentResponse with synthesized answer
        """
        # Add prompt to force answer
        messages.append(
            {
                "role": "user",
                "content": format_limit_reached_prompt(limit_type),
            }
        )

        # Get final response without tools
        kwargs: dict[str, Any] = {
            "model": self.llm_model,
            "messages": messages,  # type: ignore[arg-type]
        }
        kwargs = get_safe_llm_params(
            self.llm_model, temperature=0.0, reasoning_effort=self.reasoning_effort, **kwargs
        )

        response = self._create_chat_completion_with_retries(kwargs)

        answer = response.choices[0].message.content or ""
        retry_reason = unusable_answer_reason(answer)
        answer_retries = 0
        if retry_reason is not None:
            answer = self._retry_unusable_answer(messages, answer, retry_reason)
            answer_retries = 1
        query_time = time.time() - start_time

        return AgentResponse(
            answer=answer,
            context=context_chunks,
            metadata={
                "files_read": files_read,
                "context_sources": context_sources,
                "answer_retries": answer_retries,
                "answer_retry_reason": retry_reason,
                "llm_request_params": self._resolved_request_params(),
                "tool_calls": tool_call_count,
                "reasoning_trace": [
                    {
                        "iteration": s.iteration,
                        "tool": s.tool_name,
                        "args": s.tool_args,
                        "result_preview": s.tool_result,
                    }
                    for s in reasoning_trace
                ],
                "search_mode": search_mode.value,
                "iterations": iterations,
                "limit_reached": limit_type,
                "max_iterations_reached": limit_type == "iterations",
                "query_time": query_time,
                "prefetch_terms": prefetch["terms"],
                "prefetch_candidates": prefetch["candidates"],
                "token_usage": self.get_llm_usage(),
            },
        )

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Cache statistics dictionary
        """
        return self.cache.get_cache_stats()

    def reload_cache(self) -> bool:
        """Reload the session cache.

        Returns:
            True if reload successful
        """
        return self.cache.reload()
