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
- grep_search(pattern, path?, file_pattern?): Search for text patterns
- find_files(pattern, path?): Find files by name pattern
- get_file_info(path): Get file metadata without reading content

## Filesystem Structure
```
_meta/           → Corpus overview and navigation guide
_index/          → Topic, entity, temporal, and question indexes
  topics/        → Documents organized by subject
  entities/      → People, concepts, organizations mentioned
  temporal/      → Timeline of events
  questions/     → Query-to-document mapping
_summaries/      → Concise document summaries
documents/       → Full document content with .meta.json metadata
```

## Navigation Strategy
1. Use the cached corpus overview to understand the corpus scope
2. Based on query type:
   - For specific lookups: Check question_seeds.md or use grep_search
   - For topic exploration: Navigate topic indexes first
   - For entity queries: Check entity registry
3. Read summaries before full documents
4. Use headers_only=True for large files to get structure first
5. Read specific line ranges when you know what section you need

## Constraints
- Maximum {max_tool_calls} tool calls per query
- Maximum {max_file_reads} file reads per query
- Prefer summaries over full documents when sufficient

## Response Format
After gathering information, provide a clear, direct answer to the question without additional labels or formatting.

{strategy_hint}

## Initial Context
{initial_context}"""


# Shorter system prompt for cost-sensitive scenarios
SYSTEM_PROMPT_COMPACT = """You are a Filesystem RAG agent answering questions from a document filesystem.

Tools: list_directory, read_file, grep_search, find_files, get_file_info

Structure: _meta/ (overview), _index/ (topics, entities, questions), _summaries/, documents/

Strategy:
- Check _index/questions/question_seeds.md for direct matches
- Use _index/topics/_topic_map.md for topic queries
- Read summaries before full documents

{strategy_hint}

Context:
{initial_context}"""


# Tool usage examples for few-shot prompting
TOOL_EXAMPLES = """## Tool Usage Examples

### Example 1: Finding information about a specific topic
Question: "What are the main challenges with RAG implementations?"

Step 1 - Check question seeds:
Tool: read_file
Args: {"path": "_index/questions/question_seeds.md"}
Result: Found "What are RAG challenges?" → doc_007 (section 6), doc_012 (section 1)

Step 2 - Read summary first:
Tool: read_file
Args: {"path": "_summaries/doc_007_summary.md"}
Result: Summary shows section 6 covers "Common Challenges"

Step 3 - Read specific section:
Tool: read_file
Args: {"path": "documents/doc_007.md", "start_line": 351, "end_line": 420}
Result: Detailed content about RAG challenges

### Example 2: Looking up a specific entity
Question: "What documents mention ChromaDB?"

Step 1 - Check entity registry:
Tool: read_file
Args: {"path": "_index/entities/products.md"}
Result: ChromaDB mentioned in doc_007, doc_023

Step 2 - Read relevant summaries:
Tool: read_file
Args: {"path": "_summaries/doc_007_summary.md"}
"""


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
        compact: Use compact prompt version

    Returns:
        Formatted system prompt string
    """
    template = SYSTEM_PROMPT_COMPACT if compact else SYSTEM_PROMPT

    return template.format(
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


def format_tool_result(tool_name: str, result: str | dict, truncate_at: int = 2000) -> str:
    """Format a tool result for the conversation.

    Args:
        tool_name: Name of the tool that was called
        result: Result from the tool
        truncate_at: Maximum characters before truncation

    Returns:
        Formatted tool result string
    """
    if isinstance(result, dict):
        import json

        result_str = json.dumps(result, indent=2)
    else:
        result_str = str(result)

    # Truncate if too long
    if len(result_str) > truncate_at:
        result_str = result_str[:truncate_at] + "\n... [truncated]"

    return result_str


def format_final_answer_prompt() -> str:
    """Get prompt to request final answer from agent.

    Returns:
        Prompt string requesting final answer
    """
    return "Based on the information gathered, please provide a clear, comprehensive answer to the question."


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


# Answer extraction prompt for parsing agent responses
ANSWER_EXTRACTION_PROMPT = """Extract the following from the agent's response:

1. ANSWER: The main answer to the question
2. SOURCES: List of documents/sections cited
3. CONFIDENCE: High, Medium, or Low

Response to parse:
{response}

Format your extraction as:
ANSWER: [extracted answer]
SOURCES: [list of sources]
CONFIDENCE: [High/Medium/Low]"""


def format_answer_extraction_prompt(response: str) -> str:
    """Format prompt for extracting structured answer.

    Args:
        response: Agent's raw response

    Returns:
        Formatted extraction prompt
    """
    return ANSWER_EXTRACTION_PROMPT.format(response=response)


# Reasoning trace format
def format_reasoning_step(
    iteration: int,
    thought: str | None,
    tool_name: str | None,
    tool_args: dict | None,
    result_preview: str | None,
) -> str:
    """Format a reasoning step for logging/debugging.

    Args:
        iteration: Current iteration number
        thought: Agent's thought/reasoning (if available)
        tool_name: Name of tool called (if any)
        tool_args: Arguments passed to tool (if any)
        result_preview: Preview of result (if any)

    Returns:
        Formatted reasoning step string
    """
    lines = [f"--- Iteration {iteration} ---"]

    if thought:
        lines.append(f"Thought: {thought}")

    if tool_name:
        lines.append(f"Tool: {tool_name}")
        if tool_args:
            import json

            lines.append(f"Args: {json.dumps(tool_args)}")

    if result_preview:
        preview = result_preview[:200] + "..." if len(result_preview) > 200 else result_preview
        lines.append(f"Result: {preview}")

    return "\n".join(lines)
