# LinkedIn post

Hybrid search *lost* to plain dense retrieval on a legal benchmark — 41% vs 53% hit@5. And an agent that just reads files beat them both: 88% gold-passage access, above what the paper behind the benchmark reached with a domain-tuned legal embedder — using no embeddings at all.

The Isaacus Legal RAG Bench paper makes a claim worth testing: for legal QA, retrieval quality is the ceiling. The paper varies the embedding model and holds the architecture fixed. I wanted the other axis. Same 4,876 passages, same generator, same judge — three retrieval architectures:

- Dense vector search (Chroma, top-5)
- Hybrid dense + sparse / SPLADE (Qdrant, top-5)
- An agent that reads the corpus like a filesystem

Three findings:

1) Hybrid lost to dense twice over. It found the gold passage less often (41% vs 53%), and converted the ones it did find worse (63% vs 83% became correct answers). Legal questions barely share vocabulary with the passages that answer them, so lexical matching pulls the wrong way — and RRF wraps even good retrievals in lookalike distractors. Hybrid is a bet on lexical overlap, and legal QA is close to the worst place to make it.

2) The passive retrievers are governed by retrieval, like the paper says. Hybrid: 41% retrieval → 39% correct, 30 abstentions. Dense: 53% → 61% correct — and the 8-point overshoot is measured, not magic: the corpus restates rules across passages, and the platform's alternate-evidence signal shows dense answering correctly from non-gold passages 17 times. The ceiling is evidence-in-context; hit@5 on a single gold id is its lower bound.

3) The agent moved the ceiling. Its first full run: 59% gold access. Two rounds of trace-driven fixes later — dedupe BM25 windows, force a sibling sweep, reformulate in statutory vocabulary, require minimum evidence — 88%, same model, same corpus. Correctness followed: 82% vs dense's 61%. For a passive index, retrieval quality is a property you buy with your embedder; for an agent, it's a surface you can work.

The bill: ~30x latency (192s per question, worst case 35 minutes) and ~70x cost. In absolute terms: a cent and a half per question, $1.50 for the whole run. A search box can't pay that; a memo a lawyer will rely on doesn't notice. The real risk is elsewhere — the agent abstained zero times in 100 questions, which is great until it's confidently wrong.

All of it ran on an evaluation platform I've been building — isolated indexes, frozen config snapshots, full retrieval traces, one-click comparison export. That instrumentation caught a measurement bug that had my first run reporting 38% hit@5 against the paper's 52% (the traces proved retrieval was fine and the metric was double-counting ids). And the same traces are what took the agent from 59% to 88%. A plain script would have shipped the wrong conclusion twice.

Phase 1, disclosed honestly: cheap model, single self-judge, closed-book prompt — directional, not a replication. The number I'll defend against the paper: dense hit@5 = 53%, within one question of their 52% calibration row.

Full write-up + the platform: [link]

#RAG #LLM #AgenticAI #LegalTech #MLEngineering
