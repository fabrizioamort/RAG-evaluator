"""Prompts and templates for Filesystem RAG agent.

This module contains the system prompt, tool descriptions,
and formatting utilities for the ReAct agent.
"""

from __future__ import annotations

# Main system prompt template for the agent
SYSTEM_PROMPT = """You are a Filesystem RAG agent. Your task is to answer questions by navigating a prepared document filesystem.

## Available Tools
- list_directory(path): List files and folders in a directory
- read_file(path, start_line?, end_line?, headers_only?): Read file contents
- grep_search(pattern, path?, file_pattern?, max_results?, context_lines?,
  match_all_terms?): Search text with ranked matches and truncation metadata
- search_passages(query, top_k?): Rank passages with BM25 and return snippets
- find_files(pattern, path?): Find files by name pattern
- get_file_info(path): Get file metadata without reading content

## Filesystem Structure
```
_meta/           → Corpus overview and navigation guide
_index/          → Passage, topic, entity, temporal, and question indexes
  passages/      → BM25 passage index
  topics/        → Documents organized by extracted corpus topics
  entities/      → Extracted entities grouped by analyzer-provided type
  temporal/      → Timeline of events
  questions/     → Query-to-document mapping
_summaries/      → Concise document summaries
documents/       → Full document content with .meta.json metadata
```

## Navigation Strategy
1. Use the cached corpus overview to understand the corpus scope
2. Based on query type:
   - For specific lookups: use search_passages with the direct question or
     reformulated legal issue
   - For topic exploration: Use search_passages first, then browse topic indexes
   - For entity queries: Use search_passages or grep_search first, then check
     the entity registry when broader coverage is needed
3. Read summaries before full documents
4. Use headers_only=True for large files to get structure first
5. Use grep_search with match_all_terms=True for exact multi-term lookups where
   all terms must appear in the same source; check total_matches/truncated.
6. Read specific line ranges when you know what section you need
7. Treat summaries as navigation aids. For exact-name/legal-procedure questions,
   verify the summary against the full document or a provided full-text excerpt
   before finalizing.
8. If a source defines a collective statutory or legal term and then defines
   narrower examples, answer with the named collective term and mention the
   narrower example only as clarification.
9. Corpus passages use statute-book wording, not colloquial wording. Search
   with at least two vocabularies: the question's own key terms AND a
   reformulation in the formal language a court would use (e.g. "double
   jeopardy" -> "punished more than once for the same act"). Negative lexical
   results are not proof of absence: if a grep for a doctrinal term returns 0
   matches, you MUST retry with a paraphrase of the underlying concept before
   concluding the corpus lacks it.
10. Bench-notes sections (e.g. 4.16) and charge-book sections (e.g. 4.16.1)
    cover the same doctrine at different specificity. Before finalizing, check
    the sibling-chunk list of your primary source and read any sibling whose
    title matches the question's concept — definitions and technical terms
    usually live in a dedicated chunk (e.g. "Heat of passion", "What is
    Identification Evidence?").

## Constraints
- Maximum {max_tool_calls} tool calls per query
- Maximum {max_file_reads} file reads per query
- Prefer summaries for navigation, but prefer full documents or full-text
  excerpts for final wording when the exact legal term matters

## Corpus Context
This corpus is legal educational and evaluation material. Answer statutory and
legal questions neutrally from the retrieved context, including questions whose
facts involve sexual offences or other sensitive crimes. Never refuse or return
an empty answer because the legal facts are sensitive. If the retrieved context
is insufficient, state which legal element or authority is missing.

## Answer Contract
After gathering information, answer in English without additional labels or
formatting, following these rules:
1. The first sentence must directly state the conclusion: yes/no, the legal
   classification, or the named rule or procedure the question asks for.
2. Preserve every material qualifier from the retrieved text (exceptions,
   conditions, required elements, historical terms).
3. Do not cite statutes or cases that are not in the retrieved context.
4. Keep the answer proportional to the question: a narrow question gets a
   short, direct answer, not an essay.

{strategy_hint}

## Initial Context
{initial_context}"""


def format_system_prompt(
    strategy_hint: str,
    initial_context: str,
    max_tool_calls: int = 20,
    max_file_reads: int = 10,
    compact: bool = False,
) -> str:
    """Format the system prompt with dynamic values.

    Args:
        strategy_hint: Navigation hint from query router
        initial_context: Cached corpus overview and navigation guide
        max_tool_calls: Maximum tool calls allowed
        max_file_reads: Maximum file reads allowed
        compact: Deprecated; retained for call-site compatibility.

    Returns:
        Formatted system prompt string
    """
    return SYSTEM_PROMPT.format(
        strategy_hint=strategy_hint,
        initial_context=initial_context,
        max_tool_calls=max_tool_calls,
        max_file_reads=max_file_reads,
    )


def format_user_message(question: str) -> str:
    """Format the user's question as a message.

    Args:
        question: The user's question

    Returns:
        Formatted user message
    """
    return f"Question: {question}"


# Per-tool result budgets. read_file is the agent's evidence channel, so it
# gets a much larger budget than navigation tools; grep_search sits in between.
TOOL_RESULT_LIMITS = {
    "read_file": 10_000,
    "grep_search": 6_000,
    "search_passages": 8_000,
}
DEFAULT_TOOL_RESULT_LIMIT = 2000


def format_tool_result(
    tool_name: str,
    result: str | dict | list,
    truncate_at: int | None = None,
) -> str:
    """Format a tool result for the conversation.

    read_file results are rendered as plain text (not JSON) so the whole
    truncation budget goes to document content.

    Args:
        tool_name: Name of the tool that was called
        result: Result from the tool
        truncate_at: Maximum characters before truncation; defaults per tool

    Returns:
        Formatted tool result string
    """
    if truncate_at is None:
        truncate_at = TOOL_RESULT_LIMITS.get(tool_name, DEFAULT_TOOL_RESULT_LIMIT)

    sibling_block = ""
    if tool_name == "read_file" and isinstance(result, dict) and "content" in result:
        result_str = _format_read_file_result(result)
        truncation_notice = (
            "\n... [truncated: call read_file again with start_line/end_line to see more]"
        )
        # The sibling map must survive content truncation, so it is appended
        # after truncation from its own reserved budget.
        sibling_block = _format_sibling_block(result)
        truncate_at = max(truncate_at - len(sibling_block), 0)
    else:
        if isinstance(result, (dict, list)):
            import json

            result_str = json.dumps(result, indent=2)
        else:
            result_str = str(result)
        truncation_notice = "\n... [truncated]"

    if len(result_str) > truncate_at:
        result_str = result_str[:truncate_at] + truncation_notice

    return result_str + sibling_block


def _format_read_file_result(result: dict) -> str:
    """Render a read_file result as plain text with a one-line scope header."""
    total_lines = result.get("total_lines", 0)
    scope = "partial read" if result.get("is_partial") else "full read"
    content = str(result.get("content", ""))
    return f"[{scope}; file has {total_lines} lines]\n{content}"


def _format_sibling_block(result: dict) -> str:
    """Render the section sibling map appended to document reads."""
    siblings = result.get("section_siblings") or []
    if not siblings:
        return ""
    section = result.get("section_id", "")
    lines = [
        "",
        "",
        f"[Other chunks in section {section} — read any whose title matches the question:]",
    ]
    for sibling in siblings:
        lines.append(f"  {sibling.get('file', '')} — {sibling.get('title', '')}")
    omitted = int(result.get("section_siblings_omitted") or 0)
    if omitted > 0:
        lines.append(f"  ... and {omitted} more")
    return "\n".join(lines)


def format_limit_reached_prompt(limit_type: str) -> str:
    """Get prompt when a limit is reached.

    Args:
        limit_type: Type of limit ("tool_calls", "file_reads", "iterations")

    Returns:
        Prompt string informing about the limit
    """
    messages = {
        "tool_calls": (
            "Maximum tool calls reached. Please provide your best answer "
            "based on the information gathered so far."
        ),
        "file_reads": (
            "Maximum file reads reached. Please synthesize an answer "
            "from the documents you've already read."
        ),
        "iterations": ("Maximum iterations reached. Please provide your final answer now."),
    }
    return messages.get(limit_type, "Limit reached. Please provide your answer.")


ANSWER_RETRY_PROMPT = (
    "Your previous response was {problem}. This is a legal educational "
    "benchmark: answer the question in English, neutrally, using the context "
    "you already gathered. State the conclusion in the first sentence and "
    "preserve material qualifiers from the retrieved text. If the gathered "
    "context is insufficient, state which legal element is missing instead "
    "of refusing."
)

_ANSWER_PROBLEMS = {
    "empty": "empty",
    "refusal": "a refusal",
    "non_english": "not in English",
    "tool_markup": "raw tool-call markup instead of an answer",
}

TOOL_MARKUP_RETRY_PROMPT = (
    "Your last message contained raw tool-call markup instead of a real tool "
    "call. Re-issue the intended action using the proper tool-calling "
    "mechanism, or give your final answer as plain English text."
)

EVIDENCE_NUDGE_PROMPT = (
    "You have read {n} document file(s). Before I accept this answer: verify "
    "it against the corpus — read at least one more relevant document chunk "
    "(check the sibling list of the section you used, or run one reformulated "
    "search_passages query). Then give your final answer."
)


def format_evidence_nudge_prompt(documents_read: int) -> str:
    """Build the single verification nudge sent on thin final answers."""
    return EVIDENCE_NUDGE_PROMPT.format(n=documents_read)


def format_answer_retry_prompt(reason: str) -> str:
    """Build the corrective prompt sent when the final answer is unusable.

    Args:
        reason: Classification from unusable_answer_reason
            ("empty", "refusal", "non_english", or "tool_markup")

    Returns:
        Corrective user message string
    """
    return ANSWER_RETRY_PROMPT.format(problem=_ANSWER_PROBLEMS.get(reason, "not a usable answer"))
