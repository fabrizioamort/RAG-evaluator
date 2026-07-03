# LinkedIn post

The Isaacus Legal RAG Bench paper makes a claim worth testing: for legal QA, retrieval quality is the ceiling. Get the right passage in front of the model and it answers; miss it and nothing saves you.

The paper varies the embedding model and holds the retrieval architecture fixed. I wanted the other axis. Same 4,876 passages, same embeddings (text-embedding-3-large), same generator, same judge — and three different retrieval architectures:

- Dense vector search (Chroma, top-5)
- Hybrid dense + sparse / SPLADE (Qdrant, top-5)
- An agent that reads the corpus like a filesystem

Three things I did not fully expect:

1) Hybrid lost to plain dense. 41% vs 52% hit@5. Adding sparse made legal retrieval worse — legal questions barely share vocabulary with the passages that answer them, so lexical matching pulls the wrong way and RRF dilutes the dense signal. Hybrid is a bet on lexical overlap, and legal QA is close to the worst place to make it.

2) The passive retrievers are pinned to their ceiling exactly like the paper says. Dense: 52% retrieval → 49% correct. Hybrid: 41% → 46%. Correctness tracks retrieval almost one-to-one, and abstention is the mirror image — the worse the retrieval, the more an honest closed-book model says "I can't answer this from the context."

3) The agent climbed over the ceiling. 75% correct — sixteen points above its own gold-passage access rate (and still 66/100 counting every timeout as a failure). A top-5 retriever can't beat what's in its five chunks. An agent that reads full documents and keeps digging can. Retrieval stopped being a fixed ceiling and became a budget it could spend.

The catch: ~30x the latency (199s vs 6-9s per question), and it almost never abstains, which is great until it's confidently wrong. Fast search box → dense. A memo a lawyer will rely on → the agent earns its three minutes.

All of it ran on an evaluation platform I've been building — isolated indexes, frozen config snapshots, full retrieval traces, one-click comparison export. That instrumentation caught a measurement bug that had my first run reporting 38% hit@5 against the paper's 52%; the traces proved retrieval was fine and the metric was double-counting ids. A plain script would have shipped the wrong conclusion.

Phase 1 numbers, disclosed honestly: cheap model, single self-judge, closed-book prompt — directional, not a paper replication. The one number I'll stand behind against the paper is dense hit@5 = 52.0%, dead on their calibration row.

Full write-up + the platform: [link]

#RAG #LLM #AgenticAI #LegalTech #MLEngineering
