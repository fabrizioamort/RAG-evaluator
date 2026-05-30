# Legal RAG Bench Plan

## Goal

Use Legal RAG Bench as a repeatable showcase dataset for comparing RAG strategies in
RAG Evaluator. Start with a small subset for smoke testing, then scale to the full
corpus when the pipeline is stable.

Dataset sources:

- Blog: https://isaacus.com/blog/legal-rag-bench
- Hugging Face: https://huggingface.co/datasets/isaacus/legal-rag-bench

## Dataset Layout

Download the source files into:

```text
data/datasets/legal-rag-bench/
  corpus.jsonl
  qa.jsonl
```

Convert them with:

```powershell
uv run python scripts/convert_legal_rag_bench.py
```

The converter writes:

```text
data/legal_rag_bench/
  subset/
    raw/
    test_set.json
  full/
    raw/
    test_set.json
```

The generated data is ignored by Git. Keep the converter and this plan in source
control, but do not commit downloaded or generated dataset files.

## Smoke Test

Run the subset first to validate document loading, indexing, querying, and scoring
before spending on larger runs.

```powershell
uv run rag-eval prepare --rag-type vector_semantic --input-dir data/legal_rag_bench/subset/raw
uv run rag-eval evaluate --rag-type vector_semantic --test-set data/legal_rag_bench/subset/test_set.json --verbose
```

Then validate hybrid search:

```powershell
docker-compose up -d qdrant
uv run rag-eval prepare --rag-type vector_hybrid --input-dir data/legal_rag_bench/subset/raw
uv run rag-eval evaluate --rag-type vector_hybrid --test-set data/legal_rag_bench/subset/test_set.json --verbose
```

Optional systems:

```powershell
uv run rag-eval prepare --rag-type filesystem_rag --input-dir data/legal_rag_bench/subset/raw
uv run rag-eval evaluate --rag-type filesystem_rag --test-set data/legal_rag_bench/subset/test_set.json --verbose
```

Use Graph RAG only on a very small subset first because graph construction uses LLM
calls during indexing.

## Full Evaluation

After subset runs are stable, run Chroma and Qdrant against the full converted
dataset:

```powershell
uv run rag-eval prepare --rag-type vector_semantic --input-dir data/legal_rag_bench/full/raw
uv run rag-eval evaluate --rag-type vector_semantic --test-set data/legal_rag_bench/full/test_set.json

uv run rag-eval prepare --rag-type vector_hybrid --input-dir data/legal_rag_bench/full/raw
uv run rag-eval evaluate --rag-type vector_hybrid --test-set data/legal_rag_bench/full/test_set.json
```

Compare systems in the platform UI using the project comparison tab.

## Review Checklist

- Confirm no loader errors.
- Confirm generated answers are non-empty.
- Confirm ground-truth context is populated.
- Compare pass rate, faithfulness, relevancy, precision, recall, correctness,
  retrieval latency, and token/cost totals.
- Manually inspect a few success cases and failure cases with retrieval traces.
