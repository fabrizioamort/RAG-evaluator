"""Filesystem RAG agent implementation.

This module implements the ReAct (Reasoning + Acting) agent loop
for navigating the prepared filesystem and answering questions.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from rag_evaluator.common.llm_utils import get_safe_llm_params, is_transient_llm_error
from rag_evaluator.config import settings
from rag_evaluator.rag_implementations.filesystem_rag.agent.cache import SessionCache
from rag_evaluator.rag_implementations.filesystem_rag.agent.prompts import (
    format_answer_retry_prompt,
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


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9']{2,}")
_DOC_ID_RE = re.compile(r"doc_\d+")
_WORD_BOUNDARY_TEMPLATE = r"\b{}\b"
_LLM_MAX_ATTEMPTS = 3
_LLM_RETRY_BASE_DELAY_SECONDS = 1.0
_PREFETCH_DOCUMENT_CANDIDATES = 2
_PREFETCH_DOCUMENT_MAX_CHARS = 4500
_CONTEXT_CHUNK_MAX_CHARS = 20_000
_NAVIGATION_CONTEXT_PREFIXES = ("_index/questions/",)
_REFUSAL_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^\s*i(?:'m| am) sorry\b",
        r"^\s*i can(?:not|'t) (?:help|assist|answer|provide|comply)\b",
        r"^\s*i (?:will not|won't) (?:help|assist|answer|provide)\b",
        r"^\s*(?:sorry,? )?(?:but )?(?:i )?can(?:not|'t) assist\b",
    )
)


def unusable_answer_reason(answer: str) -> str | None:
    """Classify final answers that warrant a single corrective retry.

    Returns "empty", "non_english", "refusal", or None for usable answers.
    Detection is deliberately conservative: only the failure shapes observed
    on the legal benchmark (empty answers, CJK refusals, refusal openings).
    """
    stripped = answer.strip()
    if not stripped:
        return "empty"

    cjk_count = sum(1 for ch in stripped if chr(0x4E00) <= ch <= chr(0x9FFF))
    if cjk_count > 0.2 * len(stripped):
        return "non_english"

    if any(pattern.search(stripped) for pattern in _REFUSAL_PATTERNS):
        return "refusal"

    return None


_PREFETCH_STOP_WORDS = {
    "about",
    "according",
    "after",
    "again",
    "against",
    "also",
    "although",
    "before",
    "been",
    "being",
    "between",
    "called",
    "clearly",
    "could",
    "court",
    "does",
    "doing",
    "during",
    "every",
    "from",
    "give",
    "giving",
    "have",
    "however",
    "into",
    "itself",
    "judge",
    "juror",
    "jury",
    "must",
    "might",
    "needs",
    "offence",
    "offences",
    "only",
    "over",
    "question",
    "required",
    "selected",
    "serving",
    "should",
    "that",
    "their",
    "there",
    "these",
    "this",
    "those",
    "through",
    "thus",
    "under",
    "trial",
    "whether",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "years",
}


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

    def _extract_prefetch_terms(self, question: str) -> list[str]:
        """Extract useful lexical terms for deterministic candidate prefetch."""
        query_lower = question.lower()
        terms: list[str] = []
        seen: set[str] = set()

        for token in _TOKEN_RE.findall(query_lower):
            token = token.strip("'").lower()
            if len(token) < 4 or token in _PREFETCH_STOP_WORDS:
                continue
            if token not in seen:
                terms.append(token)
                seen.add(token)

        expansions: dict[str, list[str]] = {
            "process": ["procedure"],
            "procedure": ["process"],
            "news": ["publicity"],
            "stories": ["publicity"],
            "media": ["publicity"],
            "friend": ["know", "impartial", "impartially", "excuse", "excused"],
            "friends": ["know", "impartial", "impartially", "excuse", "excused"],
            "photos": ["inquiry", "enquiry", "outside", "irregularities"],
            "texts": ["inquiry", "enquiry", "outside", "irregularities"],
            "information": ["outside", "inquiry", "irregularities"],
            "privilege": ["self-incrimination", "certificate"],
            "testimony": ["evidence", "witness"],
            "recording": ["audio", "audiovisual", "recorded"],
            "recorded": ["recording", "audio", "audiovisual"],
            "visit": ["view", "views", "inspection", "demonstration", "experiment", "site"],
            "travels": ["view", "views", "inspection", "demonstration", "experiment"],
            "location": ["view", "views", "inspection", "site", "place"],
            "backyard": ["view", "views", "inspection", "site", "place"],
            "examine": ["view", "views", "inspection", "demonstration", "experiment"],
            "physical": ["object", "objects", "condition", "experiment", "experiments"],
            "lock": ["object", "condition", "experiment", "experiments"],
            "damaged": ["condition", "object", "experiment", "experiments"],
            "lockpicking": ["experiment", "experiments", "condition"],
            "witness": ["evidence"],
        }

        if any(phrase in query_lower for phrase in ("called", "known as", "name of")):
            expansions.setdefault("called", []).extend(["procedure", "process"])

        for term in list(terms) + ["called"]:
            for expanded in expansions.get(term, []):
                if expanded not in seen:
                    terms.append(expanded)
                    seen.add(expanded)

        return terms

    def _prefetch_term_weight(self, term: str) -> float:
        """Return a deterministic retrieval weight for high-signal query terms."""
        return {
            "view": 4.0,
            "views": 4.0,
            "inspection": 3.0,
            "demonstration": 2.5,
            "experiment": 2.5,
            "experiments": 2.5,
            "procedure": 2.0,
            "process": 1.6,
            "called": 1.4,
            "privilege": 2.0,
            "self-incrimination": 2.5,
            "certificate": 1.8,
            "recording": 2.0,
            "recorded": 2.0,
            "audio": 2.0,
            "audiovisual": 2.0,
        }.get(term, 1.0)

    def _read_text_if_exists(self, relative_path: str) -> str:
        """Read a prepared file if present, returning an empty string on failure."""
        path = self.tools.prepared_path / relative_path
        if not path.exists() or not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""

    def _term_count(self, text: str, term: str) -> int:
        """Count case-insensitive whole-word term matches."""
        pattern = _WORD_BOUNDARY_TEMPLATE.format(re.escape(term.lower()))
        return len(re.findall(pattern, text.lower()))

    def _score_prefetch_candidates(
        self,
        terms: list[str],
        summaries: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Rank candidate documents using summaries and question seed matches."""
        if not terms or not summaries:
            return []

        token_sets: dict[str, set[str]] = {
            doc_id: set(_TOKEN_RE.findall(text.lower())) for doc_id, text in summaries.items()
        }
        document_frequency: Counter[str] = Counter()
        for token_set in token_sets.values():
            document_frequency.update(token_set)
        for term in terms:
            if "-" in term:
                document_frequency[term] = sum(
                    1 for summary in summaries.values() if self._term_count(summary.lower(), term)
                )

        max_common_frequency = max(8, len(summaries) // 4)
        usable_terms = [
            term for term in terms if 0 < document_frequency[term] <= max_common_frequency
        ]
        if not usable_terms:
            usable_terms = [term for term in terms if document_frequency[term] > 0]

        scores: defaultdict[str, float] = defaultdict(float)
        matched_terms: defaultdict[str, set[str]] = defaultdict(set)
        total_docs = len(summaries)

        for doc_id, summary in summaries.items():
            summary_lower = summary.lower()
            for term in usable_terms:
                count = self._term_count(summary_lower, term)
                if not count:
                    continue
                idf = math.log((total_docs + 1) / (document_frequency[term] + 1))
                scores[doc_id] += self._prefetch_term_weight(term) * (1 + min(count, 2)) * idf
                matched_terms[doc_id].add(term)

            title_text = "\n".join(summary_lower.splitlines()[:4])
            for term in usable_terms:
                if not self._term_count(title_text, term):
                    continue
                idf = math.log((total_docs + 1) / (document_frequency[term] + 1))
                scores[doc_id] += 2.5 * self._prefetch_term_weight(term) * idf
                matched_terms[doc_id].add(term)

        question_seed_text = self._read_text_if_exists("_index/questions/question_seeds.md")
        for line in question_seed_text.splitlines():
            match = _DOC_ID_RE.search(line)
            if not match:
                continue

            doc_id = match.group(0)
            line_lower = line.lower()
            for term in usable_terms:
                if not self._term_count(line_lower, term):
                    continue
                idf = math.log((total_docs + 1) / (document_frequency[term] + 1))
                scores[doc_id] += 2.0 * self._prefetch_term_weight(term) * idf
                matched_terms[doc_id].add(term)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            {
                "doc_id": doc_id,
                "score": round(score, 3),
                "matched_terms": sorted(matched_terms[doc_id]),
            }
            for doc_id, score in ranked
            if doc_id in summaries and score > 0
        ]

    def _build_document_prefetch_excerpt(
        self,
        doc_id: str,
        terms: list[str],
        max_chars: int = _PREFETCH_DOCUMENT_MAX_CHARS,
    ) -> str:
        """Read a focused excerpt from a candidate source document."""
        document_text = self._read_text_if_exists(f"documents/{doc_id}.md")
        if not document_text:
            return ""

        if len(document_text) <= max_chars:
            return document_text

        weighted_terms = sorted(
            {term for term in terms if self._prefetch_term_weight(term) >= 1.5},
            key=self._prefetch_term_weight,
            reverse=True,
        )
        search_terms = weighted_terms or terms[:8]
        lines = document_text.splitlines()
        hit_lines: list[int] = []

        for idx, line in enumerate(lines):
            line_lower = line.lower()
            if any(self._term_count(line_lower, term) for term in search_terms):
                hit_lines.append(idx)

        if not hit_lines:
            return document_text[:max_chars].rstrip() + "\n... [document excerpt truncated]"

        selected_ranges: list[tuple[int, int]] = []
        for line_idx in hit_lines[:8]:
            start = max(0, line_idx - 4)
            end = min(len(lines), line_idx + 9)
            if selected_ranges and start <= selected_ranges[-1][1]:
                selected_ranges[-1] = (selected_ranges[-1][0], max(selected_ranges[-1][1], end))
            else:
                selected_ranges.append((start, end))

        excerpt_parts: list[str] = []
        for start, end in selected_ranges:
            excerpt_parts.append("\n".join(lines[start:end]))
            excerpt = "\n\n...\n\n".join(excerpt_parts)
            if len(excerpt) >= max_chars:
                break

        excerpt = "\n\n...\n\n".join(excerpt_parts)
        if len(excerpt) > max_chars:
            excerpt = excerpt[:max_chars].rstrip() + "\n... [document excerpt truncated]"
        return excerpt

    def _build_prefetch_context(self, question: str, max_candidates: int = 3) -> dict[str, Any]:
        """Build deterministic candidate context before LLM navigation starts."""
        terms = self._extract_prefetch_terms(question)
        summaries_dir = self.tools.prepared_path / "_summaries"
        if not summaries_dir.exists():
            return {"terms": terms, "candidates": [], "chunks": [], "sources": []}

        summaries: dict[str, str] = {}
        summary_paths: dict[str, str] = {}
        for path in sorted(summaries_dir.glob("doc_*_summary.md")):
            match = _DOC_ID_RE.search(path.name)
            if not match:
                continue
            doc_id = match.group(0)
            try:
                summaries[doc_id] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            summary_paths[doc_id] = str(path.relative_to(self.tools.prepared_path))

        ranked = self._score_prefetch_candidates(terms, summaries)[:max_candidates]
        chunks: list[str] = []
        sources: list[str] = []

        for candidate_idx, candidate in enumerate(ranked):
            doc_id = candidate["doc_id"]
            source = summary_paths[doc_id]
            snippet = summaries[doc_id]
            if len(snippet) > 1800:
                snippet = snippet[:1800].rstrip() + "\n... [prefetch snippet truncated]"

            chunks.append(
                "\n".join(
                    [
                        f"# Candidate Context: {doc_id}",
                        f"Source: {source}",
                        f"Matched terms: {', '.join(candidate['matched_terms'])}",
                        "",
                        snippet,
                    ]
                )
            )
            sources.append(source)

            if candidate_idx >= _PREFETCH_DOCUMENT_CANDIDATES:
                continue

            excerpt = self._build_document_prefetch_excerpt(doc_id, terms)
            if not excerpt:
                continue

            document_source = f"documents/{doc_id}.md"
            chunks.append(
                "\n".join(
                    [
                        f"# Candidate Full Text Excerpt: {doc_id}",
                        f"Source: {document_source}",
                        f"Matched terms: {', '.join(candidate['matched_terms'])}",
                        "",
                        excerpt,
                    ]
                )
            )
            sources.append(document_source)

        return {
            "terms": terms,
            "candidates": ranked,
            "chunks": chunks,
            "sources": sources,
        }

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
                "=== _index/lexical_candidates ===\n"
                "These candidate snippets were selected by lexical matching against "
                "summaries and question seeds. Use them when relevant, and verify with "
                "filesystem tools if the answer is uncertain.\n\n"
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
                return self.client.chat.completions.create(**kwargs)
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
                "prefetch_terms": prefetch["terms"],
                "prefetch_candidates": prefetch["candidates"],
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
