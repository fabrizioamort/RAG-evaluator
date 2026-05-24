"""Download and convert the MultiHop-RAG dataset for use with RAG Evaluator.

This script downloads the MultiHop-RAG benchmark dataset from Hugging Face
and converts it into the format expected by the RAG Evaluator project:
  - News articles → individual .txt files in data/raw/multihop_rag/
  - Multi-hop queries → test_set.json with ground truth

Dataset: https://huggingface.co/datasets/yixuantt/MultiHopRAG
Paper:   https://arxiv.org/abs/2401.15391 (COLM 2024)
License: ODC-BY (Open Data Commons Attribution License 1.0)

The corpus contains 609 real news articles (Sep-Dec 2023) from sources like
The Verge, TechCrunch, Mashable, etc. The query set contains 2,556 multi-hop
questions that require combining information from 2-4 articles.

Usage:
    # Download everything (609 articles, 2556 questions)
    uv run python scripts/load_multihop_rag.py

    # Limit to 100 questions (balanced across question types)
    uv run python scripts/load_multihop_rag.py --max-questions 100

    # Only specific question types
    uv run python scripts/load_multihop_rag.py --question-types inference comparison

    # Exclude unanswerable questions
    uv run python scripts/load_multihop_rag.py --exclude-null

    # Custom output paths
    uv run python scripts/load_multihop_rag.py --output-dir data/raw/multihop_rag --test-set data/test_set_multihop.json

    # Dry run (download and show stats, don't write files)
    uv run python scripts/load_multihop_rag.py --dry-run
"""

import argparse
import json
import re
import sys
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

# Hugging Face direct download URLs
CORPUS_URL = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json"
QUERIES_URL = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json"

# Map dataset question types to shorter category names
QUESTION_TYPE_MAP = {
    "inference_query": "inference",
    "comparison_query": "comparison",
    "temporal_query": "temporal",
    "null_query": "null",
}

# Reverse map for CLI argument parsing
QUESTION_TYPE_REVERSE = {
    "inference": "inference_query",
    "comparison": "comparison_query",
    "temporal": "temporal_query",
    "null": "null_query",
}


def download_json(url: str, label: str) -> list[dict]:
    """Download a JSON file from a URL and return parsed data."""
    print(f"  Downloading {label}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "RAG-Evaluator/1.0"})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))
        print(f"  OK - {len(data)} entries")
        return data
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)


def sanitize_filename(title: str, index: int) -> str:
    """Convert an article title into a safe filename."""
    # Remove or replace special characters
    clean = re.sub(r"[^\w\s-]", "", title)
    clean = re.sub(r"\s+", "_", clean.strip())
    clean = clean[:80]  # Truncate long titles
    if not clean:
        clean = "article"
    return f"{index:03d}_{clean}.txt"


def article_to_text(article: dict) -> str:
    """Convert a corpus article JSON object to a plain text document."""
    parts = []

    title = article.get("title", "Untitled")
    parts.append(title)
    parts.append("=" * len(title))
    parts.append("")

    # Metadata header
    if article.get("author"):
        parts.append(f"Author: {article['author']}")
    if article.get("source"):
        parts.append(f"Source: {article['source']}")
    if article.get("published_at"):
        parts.append(f"Published: {article['published_at']}")
    if article.get("category"):
        parts.append(f"Category: {article['category']}")

    parts.append("")
    parts.append("-" * 40)
    parts.append("")

    # Article body
    body = article.get("body", "")
    parts.append(body)

    return "\n".join(parts)


def build_test_case(
    query_obj: dict, case_index: int, article_title_to_file: dict[str, str]
) -> dict:
    """Convert a MultiHop-RAG query object into the project's test_case format."""
    question_type = query_obj.get("question_type", "unknown")
    category = QUESTION_TYPE_MAP.get(question_type, question_type)
    evidence_list = query_obj.get("evidence_list", [])

    # Extract ground truth context from evidence facts
    ground_truth_context = [e["fact"] for e in evidence_list if e.get("fact")]

    # Determine difficulty based on number of evidence documents needed
    num_evidence = len(evidence_list)
    if num_evidence <= 1:
        difficulty = "easy"
    elif num_evidence == 2:
        difficulty = "medium"
    else:
        difficulty = "hard"

    # Map evidence titles to source files
    source_files = []
    for e in evidence_list:
        title = e.get("title", "")
        if title in article_title_to_file:
            source_files.append(article_title_to_file[title])

    test_case = {
        "id": f"mhrag_{case_index:04d}",
        "question": query_obj["query"],
        "expected_answer": query_obj["answer"],
        "ground_truth_context": ground_truth_context,
        "difficulty": difficulty,
        "category": category,
        "requires_multihop": True,
        "question_type": question_type,
        "num_evidence_docs": num_evidence,
        "evidence_sources": source_files,
    }

    return test_case


def select_questions(
    queries: list[dict],
    max_questions: int | None,
    question_types: list[str] | None,
    exclude_null: bool,
) -> list[dict]:
    """Filter and select a subset of questions."""
    selected = queries

    # Filter by question type
    if exclude_null:
        selected = [q for q in selected if q.get("question_type") != "null_query"]

    if question_types:
        allowed = {QUESTION_TYPE_REVERSE.get(t, t) for t in question_types}
        selected = [q for q in selected if q.get("question_type") in allowed]

    # Limit count (balanced across types)
    if max_questions and len(selected) > max_questions:
        selected = _balanced_sample(selected, max_questions)

    return selected


def _balanced_sample(queries: list[dict], n: int) -> list[dict]:
    """Sample n questions balanced across question types."""
    by_type: dict[str, list[dict]] = {}
    for q in queries:
        qt = q.get("question_type", "unknown")
        by_type.setdefault(qt, []).append(q)

    num_types = len(by_type)
    per_type = n // num_types
    remainder = n % num_types

    result = []
    for i, (qt, items) in enumerate(sorted(by_type.items())):
        take = per_type + (1 if i < remainder else 0)
        take = min(take, len(items))
        result.extend(items[:take])

    return result


def print_stats(corpus: list[dict], queries: list[dict], selected: list[dict]) -> None:
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    # Corpus stats
    print(f"\nCorpus: {len(corpus)} news articles")
    sources = Counter(a.get("source", "Unknown") for a in corpus)
    categories = Counter(a.get("category", "Unknown") for a in corpus)
    print("  Sources (top 5):")
    for source, count in sources.most_common(5):
        print(f"    {source}: {count}")
    print("  Categories:")
    for cat, count in sorted(categories.items()):
        print(f"    {cat}: {count}")

    total_chars = sum(len(a.get("body", "")) for a in corpus)
    avg_chars = total_chars // len(corpus) if corpus else 0
    # Rough token estimate: 1 token ~ 4 chars for English
    est_tokens = total_chars // 4
    print(f"  Total text: ~{total_chars:,} chars (~{est_tokens:,} tokens)")
    print(f"  Avg article length: ~{avg_chars:,} chars")

    # Query stats
    print(f"\nTotal queries available: {len(queries)}")
    all_types = Counter(q.get("question_type", "unknown") for q in queries)
    for qt, count in sorted(all_types.items()):
        print(f"    {qt}: {count}")

    print(f"\nSelected queries: {len(selected)}")
    sel_types = Counter(q.get("question_type", "unknown") for q in selected)
    for qt, count in sorted(sel_types.items()):
        print(f"    {qt}: {count}")

    avg_evidence = sum(len(q.get("evidence_list", [])) for q in selected) / max(len(selected), 1)
    print(f"  Avg evidence docs per question: {avg_evidence:.1f}")

    # Cost estimate
    embed_cost = est_tokens * 0.02 / 1_000_000  # text-embedding-3-small
    query_input_tokens = len(selected) * 1500  # ~1500 tokens per query (question + context)
    query_output_tokens = len(selected) * 200   # ~200 tokens per answer
    gen_cost = (query_input_tokens * 0.15 + query_output_tokens * 0.60) / 1_000_000  # gpt-4o-mini
    eval_cost = gen_cost * 4  # LLM-as-judge uses ~4x tokens
    total_cost = embed_cost + gen_cost + eval_cost

    print(f"\n  Estimated costs (per RAG system):")
    print(f"    Embedding (text-embedding-3-small): ${embed_cost:.4f}")
    print(f"    Generation (gpt-4o-mini):           ${gen_cost:.4f}")
    print(f"    Evaluation (LLM-as-judge):          ${eval_cost:.4f}")
    print(f"    Total per RAG system:               ${total_cost:.4f}")
    print(f"    Total for 4 RAG systems:            ${total_cost * 4:.4f}")
    print(f"    (embeddings are one-time, not per-system)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and convert MultiHop-RAG dataset for RAG Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                  # All articles, all questions
  %(prog)s --max-questions 100              # 100 balanced questions
  %(prog)s --question-types inference comparison  # Only these types
  %(prog)s --exclude-null --max-questions 200     # 200 answerable questions
  %(prog)s --dry-run                        # Show stats only
        """,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/multihop_rag",
        help="Directory for converted article .txt files (default: data/raw/multihop_rag)",
    )
    parser.add_argument(
        "--test-set",
        type=str,
        default="data/test_set_multihop.json",
        help="Output path for the test set JSON (default: data/test_set_multihop.json)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to include (balanced across types)",
    )
    parser.add_argument(
        "--question-types",
        nargs="+",
        choices=["inference", "comparison", "temporal", "null"],
        default=None,
        help="Only include these question types",
    )
    parser.add_argument(
        "--exclude-null",
        action="store_true",
        help="Exclude null/unanswerable questions",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and show statistics without writing files",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MultiHop-RAG Dataset Loader for RAG Evaluator")
    print("=" * 60)
    print(f"Source: https://huggingface.co/datasets/yixuantt/MultiHopRAG")
    print(f"License: ODC-BY 1.0")
    print()

    # --- Download ---
    print("Step 1: Downloading dataset from Hugging Face...")
    corpus = download_json(CORPUS_URL, "corpus.json (news articles)")
    queries = download_json(QUERIES_URL, "MultiHopRAG.json (queries)")

    # --- Select questions ---
    selected = select_questions(
        queries, args.max_questions, args.question_types, args.exclude_null
    )

    # --- Print stats ---
    print_stats(corpus, queries, selected)

    if args.dry_run:
        print("\n[Dry run] No files written.")
        return 0

    # --- Write article files ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStep 2: Converting {len(corpus)} articles to .txt files...")
    article_title_to_file: dict[str, str] = {}

    for i, article in enumerate(corpus):
        title = article.get("title", f"article_{i}")
        filename = sanitize_filename(title, i)
        filepath = output_dir / filename
        text = article_to_text(article)
        filepath.write_text(text, encoding="utf-8")
        article_title_to_file[title] = str(filepath)

    print(f"  Written to: {output_dir}/")
    print(f"  Files created: {len(corpus)}")

    # --- Build test set ---
    print(f"\nStep 3: Building test set with {len(selected)} questions...")

    test_cases = []
    for i, query_obj in enumerate(selected):
        test_case = build_test_case(query_obj, i + 1, article_title_to_file)
        test_cases.append(test_case)

    # Collect source document paths referenced by selected questions
    referenced_files = set()
    for tc in test_cases:
        referenced_files.update(tc.get("evidence_sources", []))

    test_set = {
        "metadata": {
            "created_date": datetime.now().strftime("%Y-%m-%d"),
            "version": "1.0",
            "description": (
                "MultiHop-RAG benchmark - multi-hop queries over news articles "
                "(Sep-Dec 2023). Requires combining information from 2-4 documents."
            ),
            "source_dataset": "yixuantt/MultiHopRAG",
            "source_paper": "https://arxiv.org/abs/2401.15391",
            "license": "ODC-BY 1.0",
            "document_sources": sorted(referenced_files),
            "total_corpus_articles": len(corpus),
            "corpus_directory": str(output_dir),
        },
        "test_cases": test_cases,
    }

    test_set_path = Path(args.test_set)
    test_set_path.parent.mkdir(parents=True, exist_ok=True)
    test_set_path.write_text(json.dumps(test_set, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  Written to: {test_set_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nArticles:  {output_dir}/ ({len(corpus)} .txt files)")
    print(f"Test set:  {test_set_path} ({len(test_cases)} questions)")
    print()
    print("Next steps:")
    print(f"  1. Index the documents:")
    print(f"     uv run rag-eval prepare --rag-type vector_semantic --input-dir {output_dir}")
    print(f"     uv run rag-eval prepare --rag-type vector_hybrid --input-dir {output_dir}")
    print(f"     uv run rag-eval prepare --rag-type graph_rag --input-dir {output_dir}")
    print(f"     uv run rag-eval prepare --rag-type filesystem_rag --input-dir {output_dir}")
    print(f"  2. Run evaluation:")
    print(f"     uv run rag-eval evaluate --rag-type vector_semantic --test-set {test_set_path}")
    print(f"     uv run rag-eval evaluate --rag-type graph_rag --test-set {test_set_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
