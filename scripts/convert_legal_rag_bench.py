#!/usr/bin/env python
"""Convert Legal RAG Bench JSONL files into RAG Evaluator inputs.

Usage:
    uv run python scripts/convert_legal_rag_bench.py

Defaults:
    Input:  data/datasets/legal-rag-bench/{corpus.jsonl,qa.jsonl}
    Output: data/legal_rag_bench/{subset,full}

The converter writes every passage as a plain-text document and writes
RAG Evaluator test sets with ground-truth context pulled from the relevant
passage text.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = ROOT / "data" / "datasets" / "legal-rag-bench"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "legal_rag_bench"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into dictionaries."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object in {path} at line {line_number}")
            rows.append(row)
    return rows


def safe_filename_part(value: object) -> str:
    """Return a filesystem-safe representation of a passage id."""
    text = str(value).strip()
    safe = []
    for char in text:
        if char.isalnum() or char in {".", "-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    cleaned = "".join(safe).strip("._-")
    return cleaned or "unknown"


def passage_filename(index: int, passage_id: object) -> str:
    """Build a stable filename that also preserves the source passage id."""
    return f"passage_{index:04d}__{safe_filename_part(passage_id)}.txt"


def passage_to_text(passage: dict[str, Any]) -> str:
    """Convert a Legal RAG Bench passage row into a plain text document."""
    passage_id = str(passage.get("id", "")).strip()
    title = str(passage.get("title", "")).strip()
    text = str(passage.get("text", "")).strip()
    footnotes = str(passage.get("footnotes", "")).strip()

    parts = [
        f"Passage ID: {passage_id}",
    ]
    if title:
        parts.extend([f"Title: {title}", ""])
    if text:
        parts.append(text)
    if footnotes:
        parts.extend(["", "Footnotes:", footnotes])
    return "\n".join(parts).rstrip() + "\n"


def difficulty_for_index(index: int) -> str:
    """Assign a conservative default difficulty for Legal RAG Bench cases."""
    if index <= 25:
        return "medium"
    if index <= 75:
        return "hard"
    return "hard"


def build_test_case(
    qa: dict[str, Any],
    case_index: int,
    passage_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Convert one QA row into the RAG Evaluator test case format."""
    relevant_passage_id = str(qa.get("relevant_passage_id", "")).strip()
    passage = passage_by_id.get(relevant_passage_id)
    ground_truth = []
    if passage is not None and passage.get("text"):
        ground_truth.append(str(passage["text"]).strip())

    return {
        "id": f"legal_{case_index:03d}",
        "question": str(qa.get("question", "")).strip(),
        "expected_answer": str(qa.get("answer", "")).strip(),
        "ground_truth_context": ground_truth,
        "difficulty": difficulty_for_index(case_index),
        "category": "legal_qa",
        "relevant_passage_id": relevant_passage_id,
        "source_qa_id": qa.get("id"),
    }


def write_version(
    output_dir: Path,
    version: str,
    passages: Sequence[dict[str, Any]],
    qa_rows: Sequence[dict[str, Any]],
    passage_by_id: dict[str, dict[str, Any]],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Write one converted dataset version and return its metadata."""
    version_dir = output_dir / version
    raw_dir = version_dir / "raw"
    passage_files: dict[str, str] = {}

    test_cases = [
        build_test_case(qa, index, passage_by_id)
        for index, qa in enumerate(qa_rows, start=1)
    ]

    if not dry_run:
        if version_dir.exists():
            shutil.rmtree(version_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

        for index, passage in enumerate(passages, start=1):
            passage_id = str(passage.get("id", "")).strip()
            filename = passage_filename(index, passage_id)
            relative_path = str(Path("raw") / filename)
            passage_files[passage_id] = relative_path
            (raw_dir / filename).write_text(passage_to_text(passage), encoding="utf-8")

        document_sources = [passage_files[pid] for pid in passage_files]
        test_set = {
            "metadata": {
                "dataset": "legal-rag-bench",
                "version": version,
                "created_at": datetime.now(UTC).isoformat(),
                "passage_count": len(passages),
                "question_count": len(test_cases),
                "document_sources": document_sources,
            },
            "test_cases": test_cases,
        }
        (version_dir / "test_set.json").write_text(
            json.dumps(test_set, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return {
        "version": version,
        "passages": len(passages),
        "questions": len(test_cases),
        "path": str(version_dir),
    }


def select_subset(
    corpus: Sequence[dict[str, Any]],
    qa_rows: Sequence[dict[str, Any]],
    subset_passages: int,
    subset_questions: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a small subset whose questions all point to included passages."""
    if subset_passages < 1:
        raise ValueError("--subset-passages must be at least 1")
    if subset_questions < 1:
        raise ValueError("--subset-questions must be at least 1")

    corpus_by_id = {str(row.get("id", "")).strip(): row for row in corpus}
    selected_qa: list[dict[str, Any]] = []
    selected_ids: list[str] = []
    seen_ids: set[str] = set()

    for qa in qa_rows:
        passage_id = str(qa.get("relevant_passage_id", "")).strip()
        if passage_id not in corpus_by_id:
            continue
        selected_qa.append(qa)
        if passage_id not in seen_ids:
            selected_ids.append(passage_id)
            seen_ids.add(passage_id)
        if len(selected_qa) >= subset_questions:
            break

    if len(selected_qa) < subset_questions:
        raise ValueError(
            f"Only found {len(selected_qa)} QA rows with matching passages; "
            f"requested {subset_questions}"
        )

    for passage in corpus:
        if len(selected_ids) >= subset_passages:
            break
        passage_id = str(passage.get("id", "")).strip()
        if passage_id and passage_id not in seen_ids:
            selected_ids.append(passage_id)
            seen_ids.add(passage_id)

    selected_passages = [corpus_by_id[passage_id] for passage_id in selected_ids]
    return selected_passages, selected_qa


def validate_inputs(input_dir: Path) -> tuple[Path, Path]:
    """Return expected dataset paths, raising a clear error if missing."""
    corpus_path = input_dir / "corpus.jsonl"
    qa_path = input_dir / "qa.jsonl"
    missing = [path for path in (corpus_path, qa_path) if not path.exists()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing Legal RAG Bench input file(s): {formatted}")
    return corpus_path, qa_path


def check_relevant_passages(qa_rows: Iterable[dict[str, Any]], passage_ids: set[str]) -> list[str]:
    """Return missing relevant passage ids for QA rows."""
    missing: list[str] = []
    for qa in qa_rows:
        passage_id = str(qa.get("relevant_passage_id", "")).strip()
        if passage_id not in passage_ids:
            missing.append(passage_id)
    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Legal RAG Bench into RAG Evaluator dataset folders."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing corpus.jsonl and qa.jsonl (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for converted datasets (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--subset-passages",
        type=int,
        default=50,
        help="Number of passages to include in the subset output (default: 50)",
    )
    parser.add_argument(
        "--subset-questions",
        type=int,
        default=10,
        help="Number of questions to include in the subset output (default: 10)",
    )
    parser.add_argument(
        "--skip-subset",
        action="store_true",
        help="Do not write data/legal_rag_bench/subset",
    )
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Do not write data/legal_rag_bench/full",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print what would be written without creating files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        corpus_path, qa_path = validate_inputs(args.input_dir)
        corpus = read_jsonl(corpus_path)
        qa_rows = read_jsonl(qa_path)
        passage_by_id = {str(row.get("id", "")).strip(): row for row in corpus}
        passage_ids = set(passage_by_id)
        missing = check_relevant_passages(qa_rows, passage_ids)
        if missing:
            unique_missing = sorted(set(missing))
            raise ValueError(
                f"{len(missing)} QA rows reference missing passage ids: "
                f"{', '.join(unique_missing[:5])}"
            )

        summaries: list[dict[str, Any]] = []
        if not args.skip_subset:
            subset_passages, subset_qa = select_subset(
                corpus,
                qa_rows,
                subset_passages=args.subset_passages,
                subset_questions=args.subset_questions,
            )
            summaries.append(
                write_version(
                    args.output_dir,
                    "subset",
                    subset_passages,
                    subset_qa,
                    passage_by_id,
                    dry_run=args.dry_run,
                )
            )

        if not args.skip_full:
            summaries.append(
                write_version(
                    args.output_dir,
                    "full",
                    corpus,
                    qa_rows,
                    passage_by_id,
                    dry_run=args.dry_run,
                )
            )

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Legal RAG Bench conversion complete." if not args.dry_run else "Dry run complete.")
    for summary in summaries:
        print(
            f"  {summary['version']}: {summary['passages']} passages, "
            f"{summary['questions']} questions -> {summary['path']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
