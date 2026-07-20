# LinkedIn post

Dense vector search, hybrid, or an agent that reads files? For legal RAG I stopped guessing and measured all three — same corpus, same generator, same judge. The architecture alone moved answer accuracy from 39% to 82%.

The instrument matters as much as the result. I've been building RAG Evaluator, an open evaluation platform where every run is reproducible by construction: versioned knowledge bases, isolated indexes frozen with a full config snapshot, per-question retrieval traces, provider-reported cost per question, side-by-side comparison exports. This benchmark (Isaacus' Legal RAG Bench: 4,876 legal passages, 100 expert questions, one gold passage each) was its first full case study — and every finding below exists because of a specific platform capability.

1) The platform caught my own wrong number before I published it. First calibration run: 38% hit@5 against the paper's 52%. Because vectors and traces were stored, I could replay retrieval offline and prove the index was fine — the metric was double-counting ids. Fixed: 52%, dead on the paper's row. A plain benchmark script would have shipped the wrong conclusion with confidence.

2) A controlled comparison showed hybrid losing to plain dense: 41% vs 53% hit@5, and worse conversion of the gold passages it did find (63% vs 83%). Legal questions barely share vocabulary with the passages that answer them; sparse matching pulls the wrong way and RRF wraps good retrievals in lookalike distractors. You only see this by holding everything else fixed.

3) Retrieval traces turned the agentic RAG from mediocre to the best system I tested. First full run: 59% gold-passage access. Reading the traces exposed four concrete failure modes (duplicate BM25 windows, sibling satisficing, statutory vocabulary gaps, answering off one file read). Two fix rounds later: 88% gold access — above what the paper reached with a domain-tuned legal embedder, using no embeddings at all — and 82% correct answers, versus dense's 61%.

The bill, itemized by the platform: ~30x latency (192s per question, worst case 35 minutes) and ~70x cost — which in absolute terms is a cent and a half per question, $1.50 for the whole run. A search box can't pay that. A memo a lawyer relies on doesn't notice. And one number to watch: the agent abstained zero times in 100 questions — great, until it's confidently wrong.

Disclosed honestly: cheap model, single self-judge, closed-book prompt — Phase 1, directional, not a replication. The number I'll defend against the paper: dense hit@5 within one question of their 52%.

If you're standing at the dense-vs-hybrid-vs-agent fork: don't take my numbers. The platform and the full write-up are open — run yours.

Repo + case study: [link]

#RAG #LLM #AgenticAI #LegalTech #MLEngineering
