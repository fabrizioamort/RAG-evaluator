# Evaluation Metrics Guide

This guide explains the metrics used by the RAG Evaluator Platform to assess the quality of your RAG applications. We utilize the [DeepEval](https://github.com/confident-ai/deepeval) framework, which employs an "LLM-as-a-judge" approach to provide rigorous, semantically aware scores.

## Overview

The platform evaluates RAG systems across five distinct dimensions. You can select any combination of these metrics for each evaluation run to optimize for cost and relevance.

| Metric | What it Measures | Input Components | Key Question |
| :--- | :--- | :--- | :--- |
| **Faithfulness** | Hallucinations | Answer, Retrieved Context | Is the answer derived *only* from the context? |
| **Answer Relevancy** | Utility | Answer, Question | Does the answer actually address the question? |
| **Contextual Precision** | Ranking Quality | Question, Retrieved Context, Ground Truth | Are the most relevant chunks at the top? |
| **Contextual Recall** | Retrieval Completeness | Ground Truth Answer, Retrieved Context | Did we find *all* the necessary information? |
| **Correctness (G-Eval)** | Semantic Accuracy | Answer, Expected Answer | Is the answer factually correct? |

---

## 1. Faithfulness

**Purpose:** Measures the factual consistency of the generated answer against the retrieved context. This is your primary metric for detecting *hallucinations*.

**How it works:**
The LLM judge extracts claims from the generated answer and verifies if each claim can be inferred from the retrieved context.

- **Score:** 0.0 to 1.0 (Higher is better)
- **Low Score:** Indicates the model is inventing information not present in the source documents.

---

## 2. Answer Relevancy

**Purpose:** Measures how relevant the generated answer is to the original user question. It penalizes answers that are factual but fail to address the user's specific intent.

**How it works:**
The metric assesses the vector similarity or semantic relationship between the generated answer and the question to ensuring the response is on-topic and helpful.
*Note: This metric does not check for factual correctness, only relevance.*

- **Score:** 0.0 to 1.0 (Higher is better)
- **Low Score:** Indicates the answer is evasive, generic, or off-topic.

---

## 3. Contextual Precision

**Purpose:** Evaluates the quality of your retrieval ranking (similar to Mean Average Precision). It is critical for systems where the LLM only processes the top few chunks.

**How it works:**
It checks if the "ground truth" nodes (relevant information) appear higher in the list of retrieved context chunks. A system that retrieves relevant info in position #1 is scored higher than one that finds it in position #5.

- **Score:** 0.0 to 1.0 (Higher is better)
- **Low Score:** Indicates that while relevant info might be retrieved, it is buried under irrelevant noise, potentially confusing the LLM.

---

## 4. Contextual Recall

**Purpose:** Measures the completeness of your retrieval system.

**How it works:**
It analyzes the expected output (ground truth answer) and checks if the *retrieved context* contains all the necessary facts to generate that answer.

- **Score:** 0.0 to 1.0 (Higher is better)
- **Low Score:** Indicates the retrieval system is missing key information required to answer the question.

---

## 5. Correctness (G-Eval)

**Purpose:** Measures the semantic equivalence between the *Generated Answer* and the *Expected Answer* (Ground Truth). Unlike simple string matching (BLEU/ROUGE), this metric understands meaning.

**Criteria:**
The system uses a custom G-Eval prompt designed to:

1. Identify specific facts and entities in the expected answer.
2. Verify if the generated answer conveys the same meaning.
3. **Ignore** minor differences in formatting, punctuation, or phrasing.
4. **Penalize** contradictions or significant omissions.

- **Score:** 0.0 to 1.0 (Higher is better)
- **Low Score:** Indicates the answer is factually wrong or missing critical details compared to the ground truth.

---

## Usage Strategies

- **Debugging Hallucinations:** Focus on **Faithfulness**.
- **Tuning Retrieval:** Focus on **Contextual Precision** and **Contextual Recall**.
- **Final Acceptance Testing:** Focus on **Correctness (G-Eval)** to ensure end-to-end accuracy.
