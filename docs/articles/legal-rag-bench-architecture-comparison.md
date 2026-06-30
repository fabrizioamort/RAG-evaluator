# Retrieval is the ceiling — unless the retriever can think

*Three RAG architectures, one legal benchmark, and an evaluation platform built to tell them apart.*

Last month Isaacus published [Legal RAG Bench](https://isaacus.com/blog/legal-rag-bench) with a claim that is easy to nod along to and easy to underestimate: for legal question answering, retrieval quality is the ceiling. Put the right passage in front of the model and it usually answers well. Miss it, and no amount of model intelligence digs you out. Their numbers back it up — the best embedding model in their table tops out at 52% retrieval accuracy at k=5, and correctness rises and falls with it.

I wanted to ask something the paper doesn't. The paper varies the embedding model and holds the retrieval *architecture* fixed: FAISS over the passages, one passage per document, top-5. Sensible for an embedding study. But that is not the choice you actually make in production. You don't get to pick "the embedding model" in isolation. You pick an architecture — plain dense vector search, a hybrid dense-plus-sparse setup, or, increasingly, an agent that reads files the way a junior associate reads a binder. Same data, same embeddings, same generator, same judge. Does the *architecture* move the ceiling?

Short version: two of the three architectures landed exactly where the paper says they should, pinned to their retrieval rate. The third climbed over the ceiling and then sent me a bill for it.

Here is the headline, holding everything fixed except the architecture:

| System | Retrieval mode | Retrieval | Correct | Grounded | Abstained | Avg latency |
|---|---|---|---|---|---|---|
| vector-search (Chroma) | dense | **52.0%** hit@5 | 49.0% | 65.0% | 18 | 6.2s |
| hybrid (Qdrant + SPLADE) | dense + sparse | 41.0% hit@5 | 46.4% | 52.6% | 32 | 9.1s |
| filesystem (agent) | agentic file reads | 59.0% gold | **75.0%** | 77.3% | 2 | 198.7s |

*Retrieval metrics cover all 100 questions. Correctness and groundedness cover the cases that completed generation and judging: 100/100 for dense, 97/100 for hybrid, 88/100 for the agent (which is slow enough that a handful timed out). Phase 1 numbers; caveats are at the bottom and they matter.*

The rest of this piece is how I got those numbers, why I trust them, and the one finding that genuinely surprised me. But first I want to talk about the thing that made the experiment cheap to run and hard to fool: the platform.

## Why a platform, and not a script

I could have written this as a benchmark script. A `for` loop over 100 questions, a FAISS index, a CSV at the end. That is how most of these comparisons get done, and it is exactly why most of them are not reproducible a week later. The embedding model lives in one variable, the chunk size in another, the judge prompt in a third, and three weeks from now you cannot say with confidence which knobs were set to what when you generated the table you are about to publish.

So this ran on the evaluation platform I have been building instead, and the workflow is the point. Every run is made of persistent, inspectable entities:

- a **Knowledge Base** (the 4,876 passages, imported once, versioned — `kb_version_id` and all),
- a **RAG Config** per architecture (`vector_semantic`, `vector_hybrid`, `filesystem_rag`),
- an **isolated Index** built from that config, frozen with a full `config_snapshot`,
- a **Test Set** (the 100 expert questions, each carrying its gold `relevant_passage_id`),
- an **Evaluation** that runs a ready index against the test set and stores every retrieval trace,
- a **Comparison** that lines the finished evaluations up side by side and exports the tables in this article.

The reason that structure matters is not tidiness. It is that the snapshot is frozen at build time and travels with the result. When I export the comparison, the manifest comes with it. I do not have to remember that the hybrid index used `text-embedding-3-large` at 3,072 dimensions with `chunk_size=8000`, `chunk_overlap=0`, and `prithivida/Splade_PP_en_v1` for the sparse side. It is in the file:

```json
{
  "embedding_model": "text-embedding-3-large",
  "build_parameters": {
    "chunk_size": 8000, "chunk_overlap": 0,
    "embedding_dimension": 3072,
    "sparse_model_name": "prithivida/Splade_PP_en_v1"
  },
  "llm_model": "deepseek/deepseek-v4-flash",
  "query_execution": { "top_k": 5 },
  "rag_type": "vector_hybrid"
}
```

The other thing a platform buys you is that the same RAG classes run everywhere. The UI does not shell out to a CLI. React talks to FastAPI, FastAPI calls the same `BaseRAG` implementations the command line would. So when I compare "Chroma dense" to "the agent," I am comparing the actual retrieval code, not two re-implementations that happen to share a name.

## The benchmark, and why it bites

Legal RAG Bench is small and mean. 4,876 passages from the Victorian Criminal Charge Book. 100 questions written by people who know the material. One gold supporting passage per question, and a long-form reference answer. The official harness indexes one passage as one document — no chunking games — and measures whether the gold passage id shows up in the top 5 retrieved. That is `hit@5`, and it is the cleanest number in the whole exercise because no LLM judge touches it.

To stay honest against the paper I matched that exactly: `chunk_size=8000`, `chunk_overlap=0`, and a post-build assertion that the index contains 4,876 chunks. One passage, one chunk. If that count is not 4,876, you are measuring a different experiment and you should stop.

What makes legal QA specifically nasty is that the questions are rule-application, not keyword lookup. The corpus states a general rule; the question is a named hypothetical ("Emma is charged with..."). The right passage rarely shares vocabulary with the question. This detail is going to come back and explain almost everything below.

## The number that was wrong, and how I knew

My first calibration run reported `hit@5` of 38%. The paper says 52% for the same embedding model. A 14-point gap is not a rounding error, and it is exactly the kind of result that, published unexamined, makes you look either lucky or wrong.

The platform let me prove which. I replayed the paper's pipeline offline against the vectors already sitting in Chroma — re-embedded a stored passage and got cosine 1.0000 against its own stored vector (same embedding space, good), then ran both a brute-force cosine top-5 and Chroma's native HNSW query. Both returned 52%. Dead on the paper.

So retrieval was never the problem. The problem was my *measurement*. The id-extraction step was emitting more than one id per retrieved chunk — the real passage id, plus a synthetic doc hash, plus the raw context text — so for top-5 the five real passages were landing at list positions 1, 3, 5, 7, 9. `hit@5` checks rank ≤ 5, so it could not see the fourth and fifth passage. The fix was one id per chunk, in rank order. `hit@5` went 38% → 52%, and `gold_accessed` (a membership check, immune to the interleaving) had been right at 52% the whole time, quietly telling me the two numbers should have agreed.

I am telling this story because it is the entire argument for doing this on instrumented infrastructure. A script would have printed 38% and I would have written a worse, wronger article about how this stack underperforms the paper. The traces, the stored vectors, and the offline replay are what turned a wrong conclusion into a fixed bug.

## Setup, stated plainly

Everything below holds these fixed:

- **Corpus:** the same 4,876 passages, one Knowledge Base version, built into three isolated indexes.
- **Embeddings:** `text-embedding-3-large` at 3,072 dimensions for both vector systems.
- **Generator and judge:** `deepseek/deepseek-v4-flash` via OpenRouter, temperature 0.
- **Retrieval depth:** `top_k=5` for the vector systems.
- **Generation policy:** closed-book. The prompt licenses the model to *apply* rules from the retrieved context to the question's scenario, but if the context holds no relevant rule it abstains with a fixed sentence. This is deliberate, and it is the second thing that explains the results.

Two honest disclosures up front, because they change how you should read the table. First, the same model generated *and* judged. That is consistent across all three architectures — every run faces the identical judge — but a model grading its own homework is a known bias, so treat correctness and groundedness as directional, not as the paper's fixed GPT-5.2 verdict. Second, closed-book caps correctness near the retrieval rate by construction. That is a feature for comparing retrievers fairly and a difference from the paper, which is effectively open-book.

## Result 1: hybrid lost to plain dense

This is the one I did not expect. Conventional wisdom says dense-plus-sparse hybrid with reciprocal rank fusion is a strict upgrade over dense alone — you get semantic matching *and* exact-term matching, fused. On this corpus it went the other way: hybrid retrieved the gold passage 41% of the time against dense's 52%. Adding SPLADE made retrieval worse.

I have a theory, and I am labelling it a theory because I did not chase it all the way down. Legal questions are vocabulary-poor relative to their answers — the named hypothetical does not lexically resemble the doctrinal passage that resolves it. A sparse retriever rewards shared terms, so it happily surfaces passages that share legal vocabulary ("evidence," "jury," "charge") without sharing the specific rule, and RRF then dilutes a strong dense signal with that noisier sparse one. On a corpus where lexical overlap tracked relevance, sparse would earn its keep. Here it pulled the wrong way. The lesson I am taking is narrower and more useful than "hybrid is bad": hybrid is a bet that lexical overlap signals relevance, and legal QA is close to the worst place to make that bet.

## Result 2: the passive retrievers are pinned to their ceiling

Look at the two vector rows again. Dense: 52% retrieval, 49% correct. Hybrid: 41% retrieval, 46% correct. Correctness sits within a few points of retrieval in both cases. The paper's thesis, reproduced on my bench: passive retrieval *is* the ceiling.

The abstention column shows the mechanism. Hybrid abstained 32 times, dense 18, the agent twice. Abstention is almost the mirror image of retrieval quality — when the right passage isn't in the top 5 and the prompt is closed-book, an honest model says "I cannot answer this from the provided context" rather than confabulate. So the worst retriever (hybrid, 41%) abstains the most (32) and scores the lowest (46%). The chain is mechanical: bad retrieval → forced abstention → capped correctness. Swapping dense for hybrid did not change the model. It changed how often the model was handed the answer, and everything downstream followed.

## Result 3: the agent climbed over the ceiling

Now the filesystem RAG, and the reason this article has the title it does.

The agentic retriever does not get a top-5 budget. It treats the corpus as a filesystem — passages regrouped into documents, each with a generated summary — and it reads. It does a lexical prefetch, pulls the most promising documents, reads their full text, and decides whether it has enough. Its retrieval metric isn't `hit@5` (there is no fixed-length ranked list to score), so I report `gold_accessed`: did it actually read the gold passage. That was 59% — already better than either vector system's hit rate.

But here is the part that breaks the paper's framing. The agent's **correctness was 75%, sixteen points above its 59% gold-access rate.** A passive retriever cannot do that; its correctness is bounded by what landed in the top 5. The agent can, for two reasons. It reads *full documents*, not the lossy summaries or single chunks the vector systems live on, so when the corpus states the answer in a passage adjacent to the official gold one, it still gets there. And it almost never abstains (2 times out of 88) because it can keep digging until it finds something to stand on. Retrieval stopped being a fixed ceiling and became a budget the agent could spend.

The same property is a liability worth naming. An agent that almost never says "I don't know" is one confident-wrong answer away from a problem. Here it mostly converted abstentions into successes — 64 clean successes versus dense's 49 — but its 18 ungrounded answers are the number I would watch in production, not its accuracy.

## The cost of thinking

Nothing is free, and the agent's bill is latency. 198 seconds per question against 6 to 9 for the vector systems. Call it thirty times slower, slow enough that 12 of the 100 cases timed out before finishing. That reframes the whole comparison as an operational decision rather than a leaderboard:

- **High-volume, latency-sensitive, cost-sensitive** (a search box, an autocomplete, anything user-facing and synchronous): dense vector search. It is fast, it is cheap, and on this benchmark it matched the paper.
- **High-value, low-volume, accuracy-critical** (a memo a lawyer will actually rely on, where three minutes and a few cents are nothing against being wrong): the agent earns its latency.
- **Hybrid:** I would not reach for it on legal text after this. Sparse retrieval is a bet on lexical overlap, and this domain does not pay it out.

## What the traces showed that the table couldn't

Two qualitative finds, both surfaced by reading retrieval traces, which is the part a CSV will never give you.

The first is why I trust the agent's reading over the vector systems' chunks. One question's answer is "view" — the statute says a court may order a "demonstration, experiment or inspection," collectively called a *view*. An early agent run found the right document but only injected its *summary*, latched onto the narrower subtype, and answered "Inspection." The trace showed exactly that: right document, wrong granularity, because the summary was lossy. The fix was to inject focused full-text excerpts alongside summaries for top candidates. You cannot debug that from an accuracy column. You debug it by reading what the retriever actually put in front of the model.

The second is why correctness reads the way it does. Early on, the vector systems were abstaining even when the gold passage *was* retrieved. The trace proved it — gold at rank 2 of 5, model refused anyway — and the cause was an over-extractive prompt that read "the specific answer isn't literally in this passage" and gave up, even though the passage held the rule the question asked to apply. Rewriting the prompt to license rule-application (the closed-book policy above) fixed the false refusals without licensing open-book guessing. That distinction — abstain when the rule is absent, answer when it is present but must be applied — is the difference between a faithful legal assistant and a useless one, and it is invisible without traces.

## Caveats, because the numbers deserve them

- **Self-judge.** One model generated and judged. Consistent across architectures, but biased in absolute terms. The fixed-judge rerun (a strong, separate judge for every run) is the Phase 2 I would do before quoting these as final.
- **Closed-book policy.** It caps correctness near the retrieval rate by design and makes the vector numbers strict. The paper is effectively open-book; it recovers retrieval misses from the model's parametric legal knowledge. That gap is policy, not pipeline.
- **Phase 1 model.** `deepseek-v4-flash`, chosen for cost, is not the paper's GPT-5.2. The retrieval numbers are model-independent and calibrate cleanly; the generation numbers are directional.
- **Not a replication.** This is Legal RAG Bench used as a controlled harness to compare architectures on one stack, with the paper as a calibration reference. The one number I will stand behind against the paper is dense `hit@5` = 52.0%, which matches.
- **Incomplete agent run.** 88 of 100 agent cases completed; the rest timed out. The proportions are stable, but it is a real artifact of the latency, not a footnote I want to hide.

## What I'd change next

A fixed strong judge for every run, which is the cleanest single upgrade to credibility. A proper BM25/full-text index for the filesystem agent so its prefetch stops leaning on hand-tuned lexical weights. A reranker on the dense path, which is the obvious lever the paper leaves on the table. And an open-book toggle as a first-class config knob, so "answer from context only" versus "answer from context, fall back to your own legal knowledge" becomes a measured variable instead of a buried prompt decision.

But the finding I would lead with is the one in the title. The paper is right that retrieval is the ceiling — for retrievers that take what they're given. Hand the same corpus and the same models to a retriever that can read, reason, and read again, and the ceiling turns into a budget. It is slower, it occasionally times out, and you have to watch it for overconfidence. It is also the only one of the three that answered three out of four legal questions correctly. On the kind of question where being right is the entire point, that trade looks a lot less like a luxury and a lot more like the job.

*The numbers, config snapshots, and per-question traces in this piece were generated and exported by the evaluation platform; every table here is a direct export, manifest included.*
