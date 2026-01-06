"""Filesystem RAG agent implementation.

This module implements the ReAct (Reasoning + Acting) agent loop
for navigating the prepared filesystem and answering questions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from rag_evaluator.config import settings
from rag_evaluator.rag_implementations.filesystem_rag.agent.cache import SessionCache
from rag_evaluator.rag_implementations.filesystem_rag.agent.prompts import (
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
    ) -> None:
        """Initialize the filesystem RAG agent.

        Args:
            prepared_path: Path to the prepared filesystem root
            llm_model: OpenAI model to use for reasoning
            max_iterations: Maximum ReAct loop iterations
            max_tool_calls: Maximum total tool calls per query
            max_file_reads: Maximum file read operations per query
            client: Optional pre-configured OpenAI client
        """
        self.prepared_path = prepared_path
        self.llm_model = llm_model
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.max_file_reads = max_file_reads

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
                timeout=settings.openai_timeout,
            )

        # Warm the cache
        self.cache.warm()

    def query(self, question: str) -> AgentResponse:
        """Answer a question by navigating the filesystem.

        Args:
            question: The question to answer

        Returns:
            AgentResponse containing answer, context, and metadata
        """
        start_time = time.time()

        # Route query to determine search mode
        routing_result = self.router.route(question)
        search_mode = routing_result.mode
        strategy_hint = routing_result.strategy_hint

        # Build system prompt
        initial_context = self.cache.get_initial_context()
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
        context_chunks: list[str] = []
        tool_call_count = 0
        file_read_count = 0

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
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    # Check tool call limit
                    if tool_call_count >= self.max_tool_calls:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": format_limit_reached_prompt("tool_calls"),
                            }
                        )
                        continue

                    # Check file read limit
                    if tool_name == "read_file" and file_read_count >= self.max_file_reads:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": format_limit_reached_prompt("file_reads"),
                            }
                        )
                        continue

                    # Execute tool
                    result = self._execute_tool(tool_name, tool_args)
                    tool_call_count += 1

                    # Track file reads
                    if tool_name == "read_file":
                        file_read_count += 1
                        file_path = tool_args.get("path", "")
                        files_read.append(file_path)

                        # Store content for context
                        if isinstance(result, dict) and "content" in result:
                            context_chunks.append(result["content"])
                        elif isinstance(result, str):
                            context_chunks.append(result)

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

                query_time = time.time() - start_time

                return AgentResponse(
                    answer=answer,
                    context=context_chunks,
                    metadata={
                        "files_read": files_read,
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
                    },
                )

        # Max iterations reached - force answer
        return self._synthesize_partial_answer(
            messages=messages,
            context_chunks=context_chunks,
            reasoning_trace=reasoning_trace,
            search_mode=search_mode,
            files_read=files_read,
            tool_call_count=tool_call_count,
            start_time=start_time,
        )

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

        # Add temperature for non-reasoning models
        if not any(x in self.llm_model.lower() for x in ["o1", "o3"]):
            kwargs["temperature"] = 0

        return self.client.chat.completions.create(**kwargs)

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Execute a tool and return the result.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments for the tool

        Returns:
            Tool execution result
        """
        if tool_name == "list_directory":
            return self.tools.list_directory(**args)
        elif tool_name == "read_file":
            return self.tools.read_file(**args)
        elif tool_name == "grep_search":
            return self.tools.grep_search(**args)
        elif tool_name == "find_files":
            return self.tools.find_files(**args)
        elif tool_name == "get_file_info":
            return self.tools.get_file_info(**args)
        else:
            return f"Error: Unknown tool '{tool_name}'"

    def _synthesize_partial_answer(
        self,
        messages: list[dict[str, Any]],
        context_chunks: list[str],
        reasoning_trace: list[ReasoningStep],
        search_mode: SearchMode,
        files_read: list[str],
        tool_call_count: int,
        start_time: float,
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
                "content": format_limit_reached_prompt("iterations"),
            }
        )

        # Get final response without tools
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0 if "o1" not in self.llm_model.lower() else None,
        )

        answer = response.choices[0].message.content or ""
        query_time = time.time() - start_time

        return AgentResponse(
            answer=answer,
            context=context_chunks,
            metadata={
                "files_read": files_read,
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
                "iterations": self.max_iterations,
                "max_iterations_reached": True,
                "query_time": query_time,
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
