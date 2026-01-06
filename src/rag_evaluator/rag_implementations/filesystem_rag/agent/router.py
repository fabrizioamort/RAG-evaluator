"""Query router for Filesystem RAG agent.

This module analyzes queries to determine the optimal search strategy:
- Known-item: Direct grep + targeted reads for specific lookups
- Exploratory: Navigate indexes first for broad topic exploration
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class SearchMode(Enum):
    """Search mode for query routing."""

    KNOWN_ITEM = "known_item"
    EXPLORATORY = "exploratory"


@dataclass
class RoutingResult:
    """Result of query routing.

    Attributes:
        mode: The determined search mode
        strategy_hint: Navigation hint for the agent
        confidence: Confidence in the routing decision (0.0 to 1.0)
        matched_pattern: The pattern that triggered this routing (if any)
    """

    mode: SearchMode
    strategy_hint: str
    confidence: float
    matched_pattern: str | None = None


class QueryRouter:
    """Route queries to appropriate search strategy.

    Analyzes query text to determine whether it's a:
    - Known-item search: Looking for specific information, entity, or term
    - Exploratory search: Broad topic exploration or open-ended question

    Usage:
        router = QueryRouter()
        result = router.route("What are the main challenges with RAG?")
        print(result.mode)  # SearchMode.EXPLORATORY
        print(result.strategy_hint)  # Navigation guidance
    """

    # Patterns indicating known-item (specific lookup) queries
    KNOWN_ITEM_PATTERNS = [
        (r"where is .+ defined", 0.9),
        (r"find .+ in", 0.85),
        (r"what does .+ say about", 0.8),
        (r"look up", 0.9),
        (r"search for", 0.85),
        (r"locate .+ in", 0.9),
        (r"which (?:file|document) (?:contains|has|mentions)", 0.9),
        (r"show me .+ from", 0.8),
        (r"get .+ from", 0.8),
        (r"what is the (?:definition|meaning) of", 0.85),
        (r"who (?:is|was|are|were) .+\?", 0.7),
        (r"when (?:is|was|did)", 0.8),
        (r"where (?:is|was|are|were)", 0.8),
    ]

    # Patterns indicating exploratory (broad) queries
    EXPLORATORY_PATTERNS = [
        (r"what are (?:the )?(?:main|key|primary)", 0.9),
        (r"how does .+ work", 0.85),
        (r"explain .+", 0.9),
        (r"summarize", 0.9),
        (r"compare .+ (?:and|with|to|vs)", 0.9),
        (r"overview of", 0.9),
        (r"describe .+", 0.85),
        (r"what .+ challenges", 0.85),
        (r"what .+ benefits", 0.85),
        (r"what .+ advantages", 0.85),
        (r"what .+ disadvantages", 0.85),
        (r"tell me about", 0.8),
        (r"give me .+ overview", 0.9),
        (r"what do you know about", 0.8),
        (r"how can .+ be used", 0.85),
        (r"why (?:is|are|do|does|should)", 0.8),
        (r"what (?:is|are) .+ used for", 0.85),
    ]

    # Strategy hints for each mode
    KNOWN_ITEM_HINT = (
        "This appears to be a known-item search. "
        "Consider using grep_search directly on the query terms, "
        "or check the question_seeds.md index first for direct matches. "
        "If you know a specific document is relevant, read its summary first."
    )

    EXPLORATORY_HINT = (
        "This appears to be an exploratory query. "
        "Start by consulting the topic map (_index/topics/_topic_map.md) "
        "or entity registry (_index/entities/_entity_registry.md) "
        "to identify relevant documents before reading. "
        "Read document summaries before full documents."
    )

    DEFAULT_HINT = (
        "Query type is ambiguous. Start with the topic map to identify "
        "potentially relevant documents, then use grep_search if needed "
        "to find specific terms."
    )

    def __init__(self) -> None:
        """Initialize the query router."""
        # Compile patterns for efficiency
        self._known_item_compiled = [
            (re.compile(pattern, re.IGNORECASE), conf) for pattern, conf in self.KNOWN_ITEM_PATTERNS
        ]
        self._exploratory_compiled = [
            (re.compile(pattern, re.IGNORECASE), conf)
            for pattern, conf in self.EXPLORATORY_PATTERNS
        ]

    def route(self, query: str) -> RoutingResult:
        """Determine search mode based on query.

        Args:
            query: The user's query string

        Returns:
            RoutingResult with mode, strategy hint, and confidence
        """
        query = query.strip()

        # Check for known-item patterns
        known_item_match, known_item_conf, known_item_pattern = self._check_patterns(
            query, self._known_item_compiled
        )

        # Check for exploratory patterns
        exploratory_match, exploratory_conf, exploratory_pattern = self._check_patterns(
            query, self._exploratory_compiled
        )

        # Determine mode based on matches
        if known_item_match and not exploratory_match:
            return RoutingResult(
                mode=SearchMode.KNOWN_ITEM,
                strategy_hint=self.KNOWN_ITEM_HINT,
                confidence=known_item_conf,
                matched_pattern=known_item_pattern,
            )
        elif exploratory_match and not known_item_match:
            return RoutingResult(
                mode=SearchMode.EXPLORATORY,
                strategy_hint=self.EXPLORATORY_HINT,
                confidence=exploratory_conf,
                matched_pattern=exploratory_pattern,
            )
        elif known_item_match and exploratory_match:
            # Both matched - choose based on confidence
            if known_item_conf >= exploratory_conf:
                return RoutingResult(
                    mode=SearchMode.KNOWN_ITEM,
                    strategy_hint=self.KNOWN_ITEM_HINT,
                    confidence=known_item_conf * 0.8,  # Reduced confidence due to ambiguity
                    matched_pattern=known_item_pattern,
                )
            else:
                return RoutingResult(
                    mode=SearchMode.EXPLORATORY,
                    strategy_hint=self.EXPLORATORY_HINT,
                    confidence=exploratory_conf * 0.8,
                    matched_pattern=exploratory_pattern,
                )
        else:
            # No patterns matched - use heuristics
            return self._heuristic_route(query)

    def _check_patterns(
        self,
        query: str,
        patterns: list[tuple[re.Pattern[str], float]],
    ) -> tuple[bool, float, str | None]:
        """Check if query matches any patterns.

        Args:
            query: Query string
            patterns: List of (compiled_pattern, confidence) tuples

        Returns:
            Tuple of (matched, max_confidence, matched_pattern_str)
        """
        max_conf = 0.0
        matched_pattern: str | None = None

        for pattern, conf in patterns:
            if pattern.search(query):
                if conf > max_conf:
                    max_conf = conf
                    matched_pattern = pattern.pattern

        return max_conf > 0, max_conf, matched_pattern

    def _heuristic_route(self, query: str) -> RoutingResult:
        """Route query using heuristics when no patterns match.

        Args:
            query: Query string

        Returns:
            RoutingResult based on heuristic analysis
        """
        query_lower = query.lower()

        # Short queries are often known-item searches
        word_count = len(query.split())
        if word_count <= 3:
            return RoutingResult(
                mode=SearchMode.KNOWN_ITEM,
                strategy_hint=self.KNOWN_ITEM_HINT,
                confidence=0.6,
                matched_pattern=None,
            )

        # Questions ending with "?" are often exploratory
        if query.strip().endswith("?"):
            return RoutingResult(
                mode=SearchMode.EXPLORATORY,
                strategy_hint=self.EXPLORATORY_HINT,
                confidence=0.6,
                matched_pattern=None,
            )

        # Check for specific technical terms (more likely known-item)
        technical_terms = [
            "function",
            "class",
            "method",
            "api",
            "endpoint",
            "config",
            "setting",
            "parameter",
            "variable",
        ]
        if any(term in query_lower for term in technical_terms):
            return RoutingResult(
                mode=SearchMode.KNOWN_ITEM,
                strategy_hint=self.KNOWN_ITEM_HINT,
                confidence=0.5,
                matched_pattern=None,
            )

        # Default to exploratory (safer for general queries)
        return RoutingResult(
            mode=SearchMode.EXPLORATORY,
            strategy_hint=self.DEFAULT_HINT,
            confidence=0.5,
            matched_pattern=None,
        )

    def get_mode_string(self, mode: SearchMode) -> str:
        """Get string representation of search mode.

        Args:
            mode: SearchMode enum value

        Returns:
            String representation
        """
        return mode.value

    def get_strategy_hint(self, mode: SearchMode) -> str:
        """Get navigation hint for a specific mode.

        Args:
            mode: SearchMode enum value

        Returns:
            Strategy hint string
        """
        if mode == SearchMode.KNOWN_ITEM:
            return self.KNOWN_ITEM_HINT
        elif mode == SearchMode.EXPLORATORY:
            return self.EXPLORATORY_HINT
        else:
            return self.DEFAULT_HINT
