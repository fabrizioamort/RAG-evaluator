# Evaluation Guide

> **Master the art of RAG evaluation: from test design to result interpretation.**

This guide provides comprehensive coverage of the evaluation process, helping you design effective test sets, interpret results accurately, and systematically improve your RAG system.

---

## Table of Contents

- [Evaluation Philosophy](#evaluation-philosophy)
- [Designing Effective Test Sets](#designing-effective-test-sets)
- [Choosing the Right Metrics](#choosing-the-right-metrics)
- [Running Evaluations](#running-evaluations)
- [Interpreting Results](#interpreting-results)
- [Diagnosing Problems](#diagnosing-problems)
- [Comparing RAG Implementations](#comparing-rag-implementations)
- [Tracking Progress](#tracking-progress)
- [Best Practices](#best-practices)

---

## Evaluation Philosophy

### Why Evaluate RAG Systems?

RAG systems are complex pipelines with many components that can fail:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAG FAILURE POINTS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Documents ─▶ Chunking ─▶ Embedding ─▶ Indexing ─▶ Retrieval ─▶ Generation │
│      │           │           │           │            │             │       │
│      ▼           ▼           ▼           ▼            ▼             ▼       │
│  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────────┐ ┌─────────┐  ┌──────────┐ │
│  │Missing│  │Poor   │  │Semantic│  │Wrong      │ │Irrelevant│  │Hallucin- │ │
│  │info   │  │splits │  │drift   │  │similarity │ │context   │  │ations    │ │
│  └───────┘  └───────┘  └───────┘  └───────────┘ └─────────┘  └──────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

Evaluation helps you:
- **Identify** which component is failing
- **Quantify** the impact of changes
- **Compare** different approaches objectively
- **Prevent** regressions as you iterate

### The Evaluation Framework

Our evaluation framework uses **LLM-as-a-judge** methodology, where a powerful LLM evaluates the quality of RAG outputs against multiple criteria:

```
┌───────────────────────────────────────────────────────────────┐
│                    LLM-AS-A-JUDGE APPROACH                    │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐                    ┌─────────────────────┐  │
│  │   Inputs    │                    │      Outputs        │  │
│  │             │                    │                     │  │
│  │ • Question  │    ┌──────────┐    │ • Score (0-1)       │  │
│  │ • Answer    │───▶│   LLM    │───▶│ • Reasoning         │  │
│  │ • Context   │    │  Judge   │    │ • Evidence          │  │
│  │ • Expected  │    └──────────┘    │ • Recommendations   │  │
│  │             │                    │                     │  │
│  └─────────────┘                    └─────────────────────┘  │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## Designing Effective Test Sets

### Test Set Composition

A well-designed test set covers multiple dimensions:

![Test Set Composition](../images/eval-guide-test-composition.png)
<!-- PLACEHOLDER: eval-guide-test-composition.png - Pie chart showing ideal test set composition -->

| Category | Percentage | Purpose |
|----------|------------|---------|
| **Factual Questions** | 40% | Basic retrieval accuracy |
| **Reasoning Questions** | 25% | Multi-step inference |
| **Comparison Questions** | 15% | Cross-document analysis |
| **Negation Questions** | 10% | Testing "not mentioned" scenarios |
| **Edge Cases** | 10% | Boundary conditions |

### Question Types

#### 1. Factual Questions (What, Who, When)

```json
{
  "question": "What is the default chunk size for hybrid search?",
  "expected_answer": "700 characters",
  "difficulty": "easy",
  "tags": ["factual", "configuration"]
}
```

#### 2. Reasoning Questions (Why, How)

```json
{
  "question": "Why does hybrid search use Reciprocal Rank Fusion?",
  "expected_answer": "RRF combines results from dense and sparse search by boosting documents that rank highly in both lists, improving retrieval accuracy for queries that benefit from both semantic and keyword matching.",
  "difficulty": "medium",
  "tags": ["reasoning", "hybrid-search"]
}
```

#### 3. Comparison Questions

```json
{
  "question": "How does Graph RAG differ from Vector Semantic search?",
  "expected_answer": "Graph RAG uses a Neo4j knowledge graph to understand relationships between entities, enabling multi-hop reasoning. Vector Semantic search uses ChromaDB for similarity-based chunk retrieval without understanding relationships.",
  "difficulty": "hard",
  "tags": ["comparison", "architecture"]
}
```

#### 4. Negation Questions

```json
{
  "question": "Does the system support real-time streaming responses?",
  "expected_answer": "The documentation does not mention real-time streaming for RAG responses, though evaluation progress is streamed via SSE.",
  "difficulty": "medium",
  "tags": ["negation", "features"]
}
```

### Difficulty Levels

Assign difficulty based on retrieval complexity:

| Difficulty | Characteristics | Expected Pass Rate |
|------------|-----------------|-------------------|
| **Easy** | Single chunk, exact match | > 90% |
| **Medium** | Multiple chunks, some inference | 70-90% |
| **Hard** | Cross-document, complex reasoning | 50-70% |
| **Expert** | Edge cases, implicit information | < 50% |

### Auto-Generating Test Sets

Use the platform's AI-powered generation:

![Test Generation](../images/eval-guide-test-generation.png)
<!-- PLACEHOLDER: eval-guide-test-generation.png - Screenshot of test generation wizard -->

1. Navigate to your Test Set
2. Click **"Generate from Knowledge Base"**
3. Select the knowledge base
4. Choose difficulty distribution
5. Review and approve generated cases

```bash
# CLI alternative (not yet implemented)
uv run rag-eval generate-tests --kb-path data/indexed --output tests.json
```

---

## Choosing the Right Metrics

### Metric Selection Strategy

Not all metrics are needed for every evaluation. Choose based on your goals:

| Goal | Primary Metrics | Secondary Metrics |
|------|-----------------|-------------------|
| **Reduce hallucinations** | Faithfulness | Contextual Precision |
| **Improve retrieval** | Contextual Precision, Recall | Answer Relevancy |
| **Validate correctness** | Correctness (G-Eval) | Faithfulness |
| **Full quality check** | All five | - |

### Cost vs. Coverage Trade-offs

Each metric requires an LLM call, affecting cost and time:

| Configuration | Metrics | API Calls/Case | Cost Factor |
|---------------|---------|----------------|-------------|
| **Minimal** | Faithfulness only | 1 | 1x |
| **Balanced** | Faithfulness + Correctness | 2 | 2x |
| **Standard** | Faith + Answer Rel + Correct | 3 | 3x |
| **Comprehensive** | All 5 metrics | 5 | 5x |

### Metric Deep Dive

#### Faithfulness (Most Important)

Detects **hallucinations** - information invented by the LLM.

```
┌─────────────────────────────────────────────────────────────────┐
│                     FAITHFULNESS EVALUATION                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Context: "Paris is the capital of France. It has a            │
│           population of 2.1 million."                          │
│                                                                 │
│  Answer: "Paris is the capital of France with 2.1 million      │
│          people and is known for the Eiffel Tower."            │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Claim Analysis:                                          │   │
│  │                                                          │   │
│  │ ✓ "Paris is the capital of France" → SUPPORTED          │   │
│  │ ✓ "2.1 million people" → SUPPORTED                       │   │
│  │ ✗ "known for the Eiffel Tower" → NOT IN CONTEXT          │   │
│  │                                                          │   │
│  │ Score: 2/3 = 0.67                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Answer Relevancy

Checks if the answer **addresses the question**.

```
┌─────────────────────────────────────────────────────────────────┐
│                   ANSWER RELEVANCY EVALUATION                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Question: "How do I configure chunk size?"                     │
│                                                                 │
│  Answer A: "Set HYBRID_CHUNK_SIZE=700 in your .env file."      │
│  → Score: 0.95 (Directly answers the question)                  │
│                                                                 │
│  Answer B: "Chunk size affects retrieval quality. Larger        │
│            chunks provide more context but may include          │
│            irrelevant information."                             │
│  → Score: 0.45 (Informative but doesn't answer HOW)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Contextual Precision

Measures **ranking quality** - are relevant chunks at the top?

```
┌─────────────────────────────────────────────────────────────────┐
│                  CONTEXTUAL PRECISION EVALUATION                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Question: "What is the default embedding model?"               │
│  Ground Truth: "text-embedding-3-small"                         │
│                                                                 │
│  Retrieved Chunks (ranked):                                     │
│                                                                 │
│  1. "EMBEDDING_MODEL=text-embedding-3-small" ★ RELEVANT        │
│  2. "OpenAI provides embedding APIs..." → not relevant          │
│  3. "The system uses 1536-dimensional vectors" → not relevant   │
│                                                                 │
│  Precision@1: 1/1 = 1.0  ★ Good!                               │
│  Precision@3: 1/3 = 0.33                                        │
│                                                                 │
│  Score: Weighted average = 0.78                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Contextual Recall

Measures **retrieval completeness** - did we find everything needed?

```
┌─────────────────────────────────────────────────────────────────┐
│                   CONTEXTUAL RECALL EVALUATION                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Ground Truth: "RRF combines dense and sparse search results    │
│                using reciprocal ranking."                       │
│                                                                 │
│  Key Facts to Find:                                             │
│  • RRF (Reciprocal Rank Fusion) ✓ Found in chunk 2             │
│  • Dense search ✓ Found in chunk 1                              │
│  • Sparse search ✓ Found in chunk 1                             │
│  • Reciprocal ranking ✓ Found in chunk 2                        │
│                                                                 │
│  Score: 4/4 = 1.0 ★ Perfect recall                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Correctness (G-Eval)

Measures **semantic equivalence** with the expected answer.

```
┌─────────────────────────────────────────────────────────────────┐
│                   CORRECTNESS (G-EVAL) EVALUATION                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Expected: "The evaluation uses DeepEval framework."            │
│                                                                 │
│  Generated: "DeepEval is used for running evaluations."         │
│                                                                 │
│  G-Eval Analysis:                                               │
│  • Same core meaning? ✓ Yes                                     │
│  • All key facts present? ✓ Yes                                 │
│  • Any contradictions? ✓ No                                     │
│  • Significant omissions? ✓ No                                  │
│                                                                 │
│  Score: 0.95 (Near-perfect semantic match)                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Running Evaluations

### Pre-Evaluation Checklist

Before running an evaluation, verify:

- [ ] Knowledge base is indexed successfully
- [ ] Test set has approved test cases
- [ ] RAG configuration is selected
- [ ] API rate limits are sufficient
- [ ] Metrics are selected appropriately

### Evaluation Options

#### Via Web Platform

![Evaluation Wizard](../images/eval-guide-wizard.png)
<!-- PLACEHOLDER: eval-guide-wizard.png - Screenshot of start evaluation wizard -->

1. Click **"Start Evaluation"**
2. Select:
   - Knowledge Base (with built index)
   - Test Set
   - RAG Configuration
   - Metrics to evaluate
3. Click **"Run Evaluation"**

#### Via CLI

```bash
# Basic evaluation
uv run rag-eval evaluate --rag-type vector_semantic

# With specific test set
uv run rag-eval evaluate --rag-type vector_semantic --test-set custom_tests.json

# Verbose output
uv run rag-eval evaluate --rag-type vector_semantic --verbose

# All RAG types
uv run rag-eval evaluate --rag-type all
```

### Monitoring Progress

#### Real-Time Streaming (Platform)

![Evaluation Progress](../images/eval-guide-progress.png)
<!-- PLACEHOLDER: eval-guide-progress.png - Screenshot of evaluation progress view -->

The platform shows:
- Current test case being evaluated
- Completed vs. total cases
- Running metric scores
- Estimated time remaining
- Token usage

#### CLI Output

```
Evaluating: vector_semantic
Test Set: 25 cases
Metrics: Faithfulness, Answer Relevancy, Correctness

[1/25] What is RAG?
  → Answer: "RAG is a framework..."
  → Faithfulness: 0.95 ✓
  → Relevancy: 0.88 ✓
  → Correctness: 0.92 ✓

[2/25] How does hybrid search work?
  ...
```

---

## Interpreting Results

### Summary Dashboard

![Results Summary](../images/eval-guide-results-summary.png)
<!-- PLACEHOLDER: eval-guide-results-summary.png - Screenshot of results summary with all metric cards -->

#### Reading the Metrics

| Score Range | Interpretation | Action |
|-------------|----------------|--------|
| **0.90 - 1.00** | Excellent | Monitor for regression |
| **0.75 - 0.89** | Good | Minor tuning may help |
| **0.60 - 0.74** | Needs Improvement | Investigate specific cases |
| **Below 0.60** | Poor | Major changes needed |

### Per-Case Analysis

Click any test case to see detailed results:

![Per-Case Detail](../images/eval-guide-per-case.png)
<!-- PLACEHOLDER: eval-guide-per-case.png - Screenshot of expanded per-case result -->

#### What to Look For

1. **Low-scoring cases** - Sort by score to find problems
2. **Score variance** - High variance suggests inconsistent retrieval
3. **Metric disagreement** - High Faithfulness but low Correctness?

### Explainability Panel

![Explainability](../images/eval-guide-explainability.png)
<!-- PLACEHOLDER: eval-guide-explainability.png - Screenshot of metric explainability panel -->

The explainability panel shows the LLM judge's reasoning:

```
FAITHFULNESS ANALYSIS
─────────────────────
Claims extracted from answer:
1. "RAG combines retrieval with generation" ✓ Supported (Chunk 1, line 3)
2. "It was developed by Facebook AI" ✗ Not found in context

Verdict: 1/2 claims supported = 0.50
Recommendation: The answer includes external knowledge not present
in the retrieved context. Consider improving retrieval or adjusting
the generation prompt to stay closer to source material.
```

### Retrieval Trace

![Retrieval Trace](../images/eval-guide-trace.png)
<!-- PLACEHOLDER: eval-guide-trace.png - Screenshot of retrieval trace viewer -->

The trace shows:
- **Query embedding** time
- **Search strategy** used
- **Chunks retrieved** with scores
- **Source documents** for each chunk

---

## Diagnosing Problems

### Problem: Low Faithfulness

**Symptoms:** Answers contain information not in retrieved context

**Possible Causes:**
1. LLM using parametric knowledge
2. Insufficient context retrieved
3. Prompt not constraining output

**Solutions:**

| Cause | Solution |
|-------|----------|
| Parametric knowledge | Add system prompt: "Only use the provided context" |
| Insufficient context | Increase top_k from 5 to 10 |
| Weak prompt | Add explicit instruction to cite sources |

```python
# Example: Stricter prompt
SYSTEM_PROMPT = """Answer based ONLY on the following context.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}
"""
```

### Problem: Low Contextual Precision

**Symptoms:** Relevant information retrieved but ranked low

**Possible Causes:**
1. Chunk size too large (dilutes signal)
2. Embedding model mismatch
3. Wrong similarity metric

**Solutions:**

| Cause | Solution |
|-------|----------|
| Large chunks | Reduce from 1000 to 500 characters |
| Embedding mismatch | Try different model (e.g., text-embedding-3-large) |
| Similarity metric | Switch from cosine to dot product |

### Problem: Low Contextual Recall

**Symptoms:** Missing key information in retrieved chunks

**Possible Causes:**
1. Information spread across documents
2. top_k too low
3. Chunking splits relevant content

**Solutions:**

| Cause | Solution |
|-------|----------|
| Spread information | Try Graph RAG for multi-hop retrieval |
| Low top_k | Increase from 5 to 10-15 |
| Bad chunking | Increase chunk overlap |

### Problem: Low Answer Relevancy

**Symptoms:** Answers are factual but don't address the question

**Possible Causes:**
1. Retrieved context is off-topic
2. LLM generation drifts
3. Ambiguous questions

**Solutions:**

| Cause | Solution |
|-------|----------|
| Off-topic context | Add query rewriting |
| Generation drift | Use more focused prompt |
| Ambiguous questions | Improve test set quality |

### Problem: Low Correctness

**Symptoms:** Answers don't match expected output semantically

**Possible Causes:**
1. Incomplete retrieval
2. Generation quality
3. Expected answers too specific

**Solutions:**

| Cause | Solution |
|-------|----------|
| Incomplete retrieval | Fix recall issues first |
| Generation quality | Upgrade LLM model |
| Specific expectations | Relax expected answer wording |

---

## Comparing RAG Implementations

### Side-by-Side Comparison

![Comparison View](../images/eval-guide-comparison.png)
<!-- PLACEHOLDER: eval-guide-comparison.png - Screenshot of comparison view with multiple RAG types -->

Run evaluations on multiple RAG types and compare:

| Metric | Vector Semantic | Hybrid | Graph RAG | Filesystem |
|--------|-----------------|--------|-----------|------------|
| Faithfulness | 0.85 | 0.88 | 0.82 | 0.90 |
| Precision | 0.78 | **0.92** | 0.75 | 0.70 |
| Recall | 0.72 | 0.85 | **0.95** | 0.80 |
| Correctness | 0.80 | 0.82 | 0.78 | 0.85 |

### When to Use Each RAG Type

| Use Case | Best RAG Type | Why |
|----------|---------------|-----|
| General Q&A | Vector Semantic | Simple, fast, reliable |
| Technical docs | Hybrid | Catches keywords + concepts |
| Relationship queries | Graph RAG | Understands connections |
| Large doc sets | Filesystem | Efficient navigation |

### Baseline Comparison

Mark an evaluation as baseline to track improvements:

![Baseline Comparison](../images/eval-guide-baseline.png)
<!-- PLACEHOLDER: eval-guide-baseline.png - Screenshot of baseline comparison view -->

```
Current vs. Baseline:
─────────────────────
Faithfulness:    0.88 → 0.92  ▲ +4.5%
Precision:       0.75 → 0.82  ▲ +9.3%
Recall:          0.80 → 0.78  ▼ -2.5%
Correctness:     0.82 → 0.85  ▲ +3.7%
```

---

## Tracking Progress

### Trends Dashboard

![Trends Dashboard](../images/eval-guide-trends.png)
<!-- PLACEHOLDER: eval-guide-trends.png - Screenshot of trends dashboard with line charts -->

The Trends view shows:
- Metric scores over time
- Evaluation frequency
- Regression detection
- Improvement velocity

### Setting Up Alerts (Webhooks)

Configure webhooks to get notified:

```json
{
  "url": "https://your-server.com/webhook",
  "events": ["evaluation.completed", "evaluation.failed"],
  "threshold_alerts": {
    "faithfulness_below": 0.80,
    "correctness_below": 0.75
  }
}
```

---

## Best Practices

### Test Set Management

1. **Version your test sets** - Track changes over time
2. **Balance difficulty** - Include easy, medium, and hard questions
3. **Cover edge cases** - Test negation, multi-hop, comparison
4. **Review regularly** - Remove outdated or ambiguous cases

### Evaluation Frequency

| Stage | Frequency | Scope |
|-------|-----------|-------|
| Development | Every change | Quick (5-10 cases) |
| Pre-commit | Daily | Standard (25-50 cases) |
| Release | Weekly | Full (100+ cases) |

### Cost Optimization

1. **Use async mode** - Set `DEEPEVAL_ASYNC_MODE=True` for parallel evaluation
2. **Select metrics wisely** - Not every run needs all 5 metrics
3. **Cache results** - Don't re-evaluate unchanged configurations
4. **Use smaller models** - gpt-4o-mini is often sufficient for judging

### Documentation

Keep records of:
- Why certain configurations were chosen
- What changes improved/degraded scores
- Test set modifications and rationale

---

## Related Documentation

- [Metrics Guide](../metrics.md) - Detailed metric definitions
- [RAG Strategies Guide](../rag_strategies.md) - RAG implementation details
- [Configuration Reference](configuration.md) - All configuration options
- [Troubleshooting Guide](troubleshooting.md) - Common issues and solutions
