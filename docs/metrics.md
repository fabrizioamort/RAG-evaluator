# Evaluation Metrics

RAG Evaluator uses DeepEval-style LLM-as-judge metrics to score generated answers and
retrieved context. Scores are normalized from `0.0` to `1.0`, where higher is better.

You can run any subset of metrics in the platform evaluation wizard or through the API.
The CLI currently initializes the standard metric set.

## Metrics At A Glance

| Metric name | Measures | Inputs | Primary question |
| --- | --- | --- | --- |
| `faithfulness` | Hallucination risk | Generated answer, retrieved context | Is the answer supported by retrieved context? |
| `relevancy` | Answer usefulness | Question, generated answer | Does the answer address the question? |
| `precision` | Retrieval ranking | Question, retrieved context, expected answer | Are the most relevant chunks ranked first? |
| `recall` | Retrieval completeness | Expected answer, retrieved context | Did retrieval find the needed facts? |
| `g_eval` | End-to-end correctness | Generated answer, expected answer | Is the generated answer semantically correct? |

## Faithfulness

Faithfulness checks whether answer claims can be inferred from the retrieved context.

Use it when:

- You need to detect hallucinations.
- The answer must stay grounded in supplied documents.
- You are testing prompt changes that may cause the model to over-answer.

Low scores usually mean the generator added unsupported information, or retrieval did
not provide enough context and the model filled gaps from prior knowledge.

## Answer Relevancy

Answer relevancy checks whether the generated answer addresses the question. It does
not prove the answer is factually correct; it only checks whether it is on topic and
useful for the user intent.

Use it when:

- Answers are generic.
- The model responds to the wrong part of a question.
- Retrieved context is relevant but generation drifts.

## Contextual Precision

Contextual precision checks whether relevant chunks appear early in the retrieved
context list. This matters because most prompts privilege the first few chunks and may
ignore later evidence.

Use it when:

- Relevant facts are retrieved but buried.
- You are tuning chunk size, overlap, or top-k.
- You compare dense retrieval with hybrid search.

Low precision often points to ranking noise or oversized chunks.

## Contextual Recall

Contextual recall checks whether the retrieved context contains the facts needed to
produce the expected answer.

Use it when:

- Answers are incomplete.
- Relevant facts are spread across documents.
- You are deciding whether graph, hybrid, or agentic retrieval is needed.

Low recall often means top-k is too small, chunking split key evidence, or the chosen
retriever cannot reach the needed document.

## Correctness With G-Eval

G-Eval correctness compares the generated answer with the expected answer using a
semantic judge. It is more flexible than string matching and can tolerate equivalent
phrasing while penalizing contradictions or important omissions.

Use it when:

- You need an end-to-end acceptance metric.
- Expected answers are known.
- You want to compare final user-visible quality across RAG strategies.

## Choosing Metrics

| Goal | Recommended metrics |
| --- | --- |
| Quick smoke test | `faithfulness`, `g_eval` |
| Retrieval tuning | `precision`, `recall` |
| Hallucination reduction | `faithfulness` |
| End-to-end release check | All five metrics |
| Cost-conscious iteration | One or two primary metrics, then full metrics for candidates |

Each metric requires additional judge work, so broader metric sets cost more and take
longer. Use small test sets while iterating and full metric runs for candidate baselines.

## Interpreting Scores

| Score range | Interpretation |
| --- | --- |
| `0.90` to `1.00` | Strong result. Inspect edge cases and monitor regressions. |
| `0.75` to `0.89` | Generally usable, with targeted improvements likely. |
| `0.60` to `0.74` | Needs investigation before relying on the system. |
| Below `0.60` | Significant retrieval, generation, or test-set quality issue. |

Treat scores as decision support, not ground truth. Always inspect representative
successes and failures, especially when a change improves one metric while degrading
another.
