# Evaluation Guide

Good RAG evaluation is not only running metrics. It is designing representative test
cases, comparing systems under consistent conditions, and inspecting failures closely.

## Evaluation Workflow

1. Define the user task and success criteria.
2. Build or import a test set.
3. Build one index per RAG strategy/configuration.
4. Run evaluations against the same test set.
5. Inspect metric summaries and per-question failures.
6. Compare candidate runs against a baseline.
7. Keep the winning configuration and document the trade-offs.

## Test Set Design

A useful test set should cover the actual questions your users ask. Include:

| Question type | Purpose |
| --- | --- |
| Factual | Checks direct retrieval and exact answers. |
| Reasoning | Checks synthesis across one or more facts. |
| Comparison | Checks whether the system can distinguish entities or options. |
| Multi-hop | Checks retrieval across multiple documents or related facts. |
| Negative or missing-information cases | Checks whether the system avoids unsupported answers. |

Recommended fields:

```json
{
  "question": "What is the default embedding model?",
  "expected_answer": "The default embedding model is text-embedding-3-small.",
  "ground_truth_context": [
    "EMBEDDING_MODEL=text-embedding-3-small"
  ],
  "difficulty": "easy",
  "category": "configuration",
  "question_type": "factual"
}
```

`ground_truth_context` improves contextual metric quality. Keep it concise and relevant.

## Test Set Size

| Stage | Suggested size | Purpose |
| --- | --- | --- |
| Smoke test | 3 to 10 cases | Validate setup, indexes, and provider configuration. |
| Development iteration | 10 to 30 cases | Catch common failures quickly. |
| Candidate comparison | 30 to 100 cases | Compare strategies with less noise. |
| Release or regression suite | 100+ cases | Monitor production-quality changes over time. |

Quality matters more than size. Remove ambiguous, outdated, or unverifiable cases.

## Metric Selection

| Goal | Metrics |
| --- | --- |
| Detect hallucinations | `faithfulness` |
| Tune retrieval ranking | `precision` |
| Tune retrieval coverage | `recall` |
| Check final answer quality | `g_eval` |
| Confirm answer usefulness | `relevancy` |

For early runs, start with `faithfulness` and `g_eval`. Add `precision` and `recall`
when tuning retrieval. Run all metrics for final comparisons.

## Running Evaluations

### Web Platform

1. Build a ready index.
2. Open the project's Evaluations tab.
3. Start an evaluation.
4. Select the index, test set, and metrics.
5. Watch streamed progress.
6. Review the completed results.

### CLI

```powershell
uv run rag-eval prepare --rag-type vector_semantic --input-dir data/raw
uv run rag-eval evaluate --rag-type vector_semantic --test-set data/test_set.json --verbose
```

Evaluate all registered RAG types:

```powershell
uv run rag-eval evaluate --rag-type all --test-set data/test_set.json
```

## Interpreting Results

Start at the aggregate view, then inspect individual cases.

Ask:

- Which metric is weak?
- Are failures concentrated in one category or difficulty?
- Did retrieval miss the source text?
- Did retrieval find the source but rank it low?
- Did generation ignore or contradict retrieved context?
- Is the expected answer too strict or ambiguous?

Common patterns:

| Symptom | Likely cause | Next step |
| --- | --- | --- |
| Low faithfulness | Generator adds unsupported facts | Tighten prompt or inspect retrieved context. |
| Low precision | Relevant chunks are buried | Tune chunking, top-k, or try hybrid search. |
| Low recall | Key facts are absent | Increase top-k, add documents, or try graph/agentic retrieval. |
| Low relevancy | Answer does not address the question | Improve prompt or query handling. |
| Low G-Eval with high faithfulness | Answer is grounded but incomplete/wrong | Inspect retrieval completeness and expected answer. |

## Comparisons

Compare systems only when the inputs are consistent:

- Same project.
- Same knowledge base content.
- Same test set.
- Same metric list.
- One major change at a time when possible.

Use the Comparisons tab to choose a completed baseline evaluation and one or more
completed alternatives. Review aggregate deltas and per-question deltas. A strategy
that improves average score but fails critical questions may not be the right choice.

## Baselines And Trends

Mark a completed evaluation as the project baseline when it represents the current
accepted configuration. Future runs can then be interpreted as regressions or
improvements.

Use Trends to observe:

- Metric changes over time.
- Pass-rate movement.
- Cost and token changes.
- Latency changes.

## Cost Control

- Start with small test sets.
- Run fewer metrics during iteration.
- Use cheaper judge/generation models for exploratory runs when acceptable.
- Keep `DEEPEVAL_ASYNC_MODE=False` if you are hitting provider rate limits.
- Inspect playground results before running large evaluations.

## Review Checklist

Before trusting a result:

- Confirm the index status was `ready`.
- Confirm the test set cases were reviewed.
- Confirm metric selection matches the decision you are making.
- Inspect several high-scoring and low-scoring cases manually.
- Check retrieval traces for representative failures.
- Compare cost and latency, not only quality.
- Save the rationale for baseline changes in the baseline reason or notes.
