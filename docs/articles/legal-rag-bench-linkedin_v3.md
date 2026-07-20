# LinkedIn post v3

I built a RAG evaluation platform, then used Isaacus' Legal RAG Bench to test the architecture question teams usually answer by instinct:

dense vector search, hybrid retrieval, or an agent that reads?

Same benchmark. Same 100 legal questions. Same generator. Same judge. Different architecture.

The result was not just a leaderboard. It was a product test: could RAG Evaluator make the decision reproducible, inspectable, and useful enough to debug?

Legal RAG Bench is a published legal QA benchmark: 4,876 passages from the Victorian Criminal Charge Book, 100 expert-written questions, gold passage labels, and a paper arguing that retrieval quality sets the ceiling for legal QA.

I used it as a calibration harness for three RAG architectures:

- Chroma dense vector search: 53% hit@5, 61% correct
- Qdrant + SPLADE hybrid retrieval: 41% hit@5, 39% correct
- Filesystem/agentic RAG: 88% gold access, 82% correct

The platform mattered as much as the numbers.

RAG Evaluator turns the workflow into managed, inspectable entities: versioned Knowledge Bases, RAG Configs, isolated Indexes, Test Sets, Evaluations, retrieval traces, cost/latency tracking, side-by-side Comparisons, Trends, and exportable manifests.

That infrastructure changed the experiment in three concrete ways.

1. It caught a wrong number before I published it.

My first dense run reported 38% hit@5. The paper baseline was 52%. Because the vectors, traces, and per-question retrievals were stored, I replayed retrieval offline and proved the index was fine. The metric was wrong: id extraction was double-counting retrieved chunks. Fixed result: 52%, matching the paper.

An evaluation you cannot audit is an opinion with decimals.

2. Hybrid retrieval lost to dense retrieval on this corpus.

Legal rule-application questions often do not share vocabulary with the passages that answer them. The sparse side pulled in lexically similar distractors and diluted the dense signal. The lesson is not "hybrid is bad". The lesson is narrower and more useful: hybrid is a bet that lexical overlap signals relevance. On this legal QA corpus, that bet did not pay.

3. The agent improved through trace-driven engineering, not model changes.

The first full agent run reached 59% gold access. Reading traces exposed duplicate BM25 windows, sibling-passage satisficing, vocabulary gaps, and answers emitted after too little evidence. Two engineering rounds later: 88% gold access and 82% correct answers.

That is in the same retrieval-access range as the paper's domain-tuned legal embedder, though it is not an apples-to-apples metric comparison: the agent's gold access is not rank-limited hit@5. Directionally, the point is still important. The ceiling moved because the retriever could read, inspect traces, and be improved.

The bill was real: about 192 seconds per question for the agent versus 6-7 seconds for the vector systems, and roughly 70x the cost per question. In absolute terms, though, that was about $0.015 per question: $1.50 for the whole 100-question run on a cheap model.

A search box cannot pay that latency. A legal memo someone will rely on might.

Disclosed honestly: this is Phase 1, not a replication. The generation numbers use a cheap model, a single self-judge, and a closed-book prompt. The dense retrieval calibration is the number I would put directly against the paper: 52-53% hit@5 against their 52%.

The takeaway is not that everyone should use agentic RAG.

The takeaway is that architecture choices should leave evidence behind: corpus version, config snapshots, retrieval traces, prompts, judge settings, costs, latencies, and comparisons you can audit later.

If you are standing at the dense-vs-hybrid-vs-agent fork, do not take my numbers. Run the experiment on your corpus.

Full write-up + repo: [link]

Disclosure: drafted with AI assistance, fully reviewed, fact-checked, and edited by me. Technical decisions, analysis, and conclusions are mine. This is a software evaluation case study, not legal advice.

#RAG #LLM #AgenticAI #LegalTech #MLEngineering #GenAI

Suggested image to attach: `docs/images/legal-rag-retrieval-vs-correctness-v1.png`
