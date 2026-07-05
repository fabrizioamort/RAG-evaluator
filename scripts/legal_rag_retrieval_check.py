#!/usr/bin/env python
"""Check Legal RAG Bench gold-passage ranks in a prepared BM25 index.

This is a retrieval-only regression harness for Filesystem RAG. It makes no
LLM calls; it loads the prepared ``_index/passages/bm25.json`` file and reports
where each locked Legal RAG Bench gold passage appears in BM25 results.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_evaluator.rag_implementations.filesystem_rag.agent.prefetch import (  # noqa: E402
    build_prefetch_context,
)
from rag_evaluator.rag_implementations.filesystem_rag.agent.tools import (  # noqa: E402
    FilesystemRAGTools,
)
from rag_evaluator.rag_implementations.filesystem_rag.passage_index import (  # noqa: E402
    BM25_INDEX_RELATIVE_PATH,
    BM25PassageIndex,
)

DEFAULT_PREPARED_PATH = ROOT / "data" / "prepared" / "filesystem_rag"
DEFAULT_TOP_K = 30


@dataclass(frozen=True)
class RetrievalCase:
    """One retrieval-check case."""

    case_id: str
    name: str
    question: str
    gold_passage_id: str


LOCKED_CASES: tuple[RetrievalCase, ...] = (
    RetrievalCase(
        case_id="legal_001",
        name="Bob & Ted",
        question=(
            "Bob and Ted are close friends. Ted is on trial for drug offences, and Bob has "
            "been selected as a juror in Ted's case. Is the judge required to excuse Bob "
            "from serving on the jury?"
        ),
        gold_passage_id="1.2-c2-s2",
    ),
    RetrievalCase(
        case_id="legal_002",
        name="Harry",
        question=(
            "Harry is serving as a juror in a burglary trial. He is also a professional "
            "locksmith. During the proceedings, Harry is shown a lock which, in his expert "
            "view, is too damaged to have been lockpicked in the manner described in court. "
            "Why might Harry's expert knowledge of lockpicking be irrelevant when assessing "
            "the physical evidence?"
        ),
        gold_passage_id="1.5-c5-s1",
    ),
    RetrievalCase(
        case_id="legal_003",
        name="Isaac",
        question=(
            "Isaac is on trial for statutory murder of an emergency worker. You are his "
            "barrister. Should you notify the jury that, according to the Crimes Act 1958, "
            "the standard sentence for this offence is 30 years imprisonment?"
        ),
        gold_passage_id="1.5-c6-s1",
    ),
    RetrievalCase(
        case_id="legal_004",
        name="Juror News Stories",
        question=(
            "Should jurors be excused if they have encountered news stories about the "
            "accused prior to the trial commencing?"
        ),
        gold_passage_id="1.5-c7-s2",
    ),
    RetrievalCase(
        case_id="legal_005",
        name="Frank & Joe",
        question=(
            "Frank and Joe are jurors in an arson trial. Over the weekend, Joe finds photos "
            "of the accused holding a petrol canister and texts them to Frank. Having "
            "received this new information, what should Frank do?"
        ),
        gold_passage_id="1.5-c8-s1",
    ),
    RetrievalCase(
        case_id="legal_006",
        name="Reasonable Doubt",
        question=(
            "Does the standard of proof of \"beyond reasonable doubt\" imply that the jury "
            "must be fully convinced of every claim the prosecution has made?"
        ),
        gold_passage_id="1.7-c3-s2",
    ),
    RetrievalCase(
        case_id="legal_007",
        name="Sally View",
        question=(
            "Sally is accused of cultivating narcotic plants in her backyard. One of the "
            "elements of this charge is that \"the accused intentionally cultivated or "
            "attempted to cultivate a particular substance.\" To establish whether this is "
            "the case, the judge believes it would be valuable to visit Sally's backyard "
            "and have the jury examine it for themselves. What is the name of the legal "
            "procedure whereby the court travels to a location relevant to the charge?"
        ),
        gold_passage_id="2.1-c1-s1",
    ),
    RetrievalCase(
        case_id="legal_008",
        name="Josh",
        question=(
            "Josh is 21 years old. He witnessed a murder that was clearly perpetrated by "
            "the accused. The prosecution, however, needs to determine whether the accused "
            "committed the acts voluntarily, thus satisfying the elements of intentional or "
            "reckless murder. In court, the judge plays a recording of Josh giving his "
            "eyewitness testimony. What is this evidentiary process called?"
        ),
        gold_passage_id="2.3.3-c2-s1",
    ),
    RetrievalCase(
        case_id="legal_009",
        name="Olivia",
        question=(
            "Olivia illegally bought cannabis from a tobacconist. As she left, she saw a "
            "man firebomb the store. She is now a witness in the arson trial. What legal "
            "privilege should she consider when giving her evidence?"
        ),
        gold_passage_id="2.5-c1-s1",
    ),
    RetrievalCase(
        case_id="legal_010",
        name="Emma",
        question=(
            "Emma is found in possession of someone else's phone but has no documentation "
            "of how she obtained it. Emma's counsel argues that possession of the phone is "
            "circumstantial evidence and thus completely inadmissible to establish guilt. "
            "Is Emma's counsel correct?"
        ),
        gold_passage_id="3.6-c2-s1",
    ),
)


def load_cases(path: Path | None) -> list[RetrievalCase]:
    """Load retrieval cases from a RAG Evaluator test_set JSON or use locked defaults."""
    if path is None:
        return list(LOCKED_CASES)

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("test_cases", payload)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list or test_set object in {path}")

    cases: list[RetrievalCase] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Expected object for case {index} in {path}")
        question = str(row.get("question", "")).strip()
        gold = str(
            row.get("relevant_passage_id")
            or row.get("metadata", {}).get("relevant_passage_id")
            or ""
        ).strip()
        if not question or not gold:
            raise ValueError(f"Case {index} in {path} is missing question or relevant_passage_id")
        cases.append(
            RetrievalCase(
                case_id=str(row.get("id") or f"case_{index:03d}"),
                name=str(row.get("name") or row.get("id") or f"Case {index}"),
                question=question,
                gold_passage_id=gold,
            )
        )
    return cases


def result_matches_gold(result: dict[str, Any], gold_passage_id: str) -> bool:
    """Return True if a BM25 result identifies the requested gold passage."""
    passage_id = str(result.get("passage_id") or "")
    doc_id = str(result.get("doc_id") or "")
    source = str(result.get("source") or "")
    source_stem = Path(source).stem if source else ""
    return (
        doc_id == gold_passage_id
        or source_stem == gold_passage_id
        or passage_id == gold_passage_id
        or passage_id.startswith(f"{gold_passage_id}#")
    )


def evaluate_case(
    index: BM25PassageIndex,
    case: RetrievalCase,
    top_k: int,
) -> dict[str, Any]:
    """Search one case and return rank metadata."""
    search_result = index.search(case.question, top_k=top_k)
    results = list(search_result.get("results", []))
    rank: int | None = None
    gold_result: dict[str, Any] | None = None
    for position, result in enumerate(results, start=1):
        if result_matches_gold(result, case.gold_passage_id):
            rank = position
            gold_result = result
            break

    top_result = results[0] if results else {}
    return {
        "case_id": case.case_id,
        "name": case.name,
        "gold_passage_id": case.gold_passage_id,
        "rank": rank,
        "in_top_5": rank is not None and rank <= 5,
        "in_top_k": rank is not None,
        "top_doc_id": top_result.get("doc_id", ""),
        "top_passage_id": top_result.get("passage_id", ""),
        "top_score": top_result.get("score", ""),
        "gold_score": gold_result.get("score", "") if gold_result else "",
        "matched_terms": " ".join(gold_result.get("matched_terms", [])) if gold_result else "",
    }


def evaluate_prefetch_case(
    tools: FilesystemRAGTools,
    case: RetrievalCase,
    top_k: int,
) -> dict[str, Any]:
    """Search one case through the deterministic prefetch pipeline."""
    prefetch = build_prefetch_context(tools, case.question, max_candidates=top_k)
    results = list(prefetch.get("candidates", []))
    rank: int | None = None
    gold_result: dict[str, Any] | None = None
    for position, result in enumerate(results, start=1):
        if result_matches_gold(result, case.gold_passage_id):
            rank = position
            gold_result = result
            break

    top_result = results[0] if results else {}
    return {
        "case_id": case.case_id,
        "name": case.name,
        "gold_passage_id": case.gold_passage_id,
        "rank": rank,
        "in_top_5": rank is not None and rank <= 5,
        "in_top_k": rank is not None,
        "top_doc_id": top_result.get("doc_id", ""),
        "top_passage_id": top_result.get("passage_id", ""),
        "top_score": top_result.get("score", ""),
        "gold_score": gold_result.get("score", "") if gold_result else "",
        "matched_terms": " ".join(gold_result.get("matched_terms", [])) if gold_result else "",
    }


def summarize(rows: list[dict[str, Any]], top_k: int) -> dict[str, int]:
    """Return aggregate retrieval counts."""
    return {
        "cases": len(rows),
        "top_5": sum(1 for row in rows if row["in_top_5"]),
        f"top_{top_k}": sum(1 for row in rows if row["in_top_k"]),
        f"missing_top_{top_k}": sum(1 for row in rows if not row["in_top_k"]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-case results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "name",
        "gold_passage_id",
        "rank",
        "in_top_5",
        "in_top_k",
        "top_doc_id",
        "top_passage_id",
        "top_score",
        "gold_score",
        "matched_terms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_report(rows: list[dict[str, Any]], summary: dict[str, int], top_k: int) -> None:
    """Print a compact human-readable report."""
    print("Legal RAG retrieval check")
    print(
        f"cases={summary['cases']} top_5={summary['top_5']} "
        f"top_{top_k}={summary[f'top_{top_k}']} "
        f"missing_top_{top_k}={summary[f'missing_top_{top_k}']}"
    )
    print()
    print(f"{'case':<10} {'name':<20} {'gold':<14} {'rank':<8} {'top_doc'}")
    print("-" * 76)
    for row in rows:
        rank = row["rank"] if row["rank"] is not None else f">{top_k}"
        print(
            f"{row['case_id']:<10} {row['name'][:20]:<20} "
            f"{row['gold_passage_id']:<14} {str(rank):<8} {row['top_doc_id']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank Legal RAG Bench gold passages with the prepared BM25 index."
    )
    parser.add_argument(
        "--prepared-path",
        type=Path,
        default=DEFAULT_PREPARED_PATH,
        help=f"Prepared filesystem root (default: {DEFAULT_PREPARED_PATH})",
    )
    parser.add_argument(
        "--cases-json",
        type=Path,
        default=None,
        help="Optional RAG Evaluator test_set JSON; defaults to the locked 10-case subset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Rank depth to inspect (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--mode",
        choices=("bm25", "prefetch"),
        default="bm25",
        help="Retrieval path to evaluate: raw BM25 baseline or Phase-2 prefetch.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--csv-out", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--expect-top-5", type=int, default=None, help="Fail unless top-5 count matches.")
    parser.add_argument(
        "--expect-missing-top-k",
        type=int,
        default=None,
        help="Fail unless missing-at-top-k count matches.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        print("ERROR: --top-k must be at least 1", file=sys.stderr)
        return 2

    index_path = args.prepared_path / BM25_INDEX_RELATIVE_PATH
    if not index_path.exists():
        print(
            f"ERROR: BM25 index not found at {index_path}. "
            "Pass --prepared-path for the locked prepared index or re-run preparation.",
            file=sys.stderr,
        )
        return 2

    try:
        cases = load_cases(args.cases_json)
        if args.mode == "bm25":
            index = BM25PassageIndex.load(args.prepared_path)
            rows = [evaluate_case(index, case, args.top_k) for case in cases]
        else:
            tools = FilesystemRAGTools(str(args.prepared_path))
            rows = [evaluate_prefetch_case(tools, case, args.top_k) for case in cases]
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = summarize(rows, args.top_k)
    if args.csv_out is not None:
        write_csv(args.csv_out, rows)

    if args.json:
        print(json.dumps({"summary": summary, "results": rows}, indent=2))
    else:
        print_report(rows, summary, args.top_k)

    expected_failures: list[str] = []
    if args.expect_top_5 is not None and summary["top_5"] != args.expect_top_5:
        expected_failures.append(f"top_5 expected {args.expect_top_5}, got {summary['top_5']}")
    missing_key = f"missing_top_{args.top_k}"
    if (
        args.expect_missing_top_k is not None
        and summary[missing_key] != args.expect_missing_top_k
    ):
        expected_failures.append(
            f"{missing_key} expected {args.expect_missing_top_k}, got {summary[missing_key]}"
        )
    if expected_failures:
        print("ERROR: " + "; ".join(expected_failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
