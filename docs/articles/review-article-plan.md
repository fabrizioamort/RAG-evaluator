# Article Review Plan — Legal RAG Bench comparison

Living document tracking the review, fixes, and publication plan for:

- `docs/articles/legal-rag-bench-architecture-comparison.md` (long-form, GitHub)
- `docs/articles/legal-rag-bench-linkedin.md` (short, LinkedIn)

Review date: 2026-07-02. Update statuses here as the articles evolve.

## Verified facts (do not re-verify)

Checked against the Isaacus paper ([blog](https://isaacus.com/blog/legal-rag-bench),
[arXiv 2603.01710](https://arxiv.org/abs/2603.01710),
[benchmark repo](https://github.com/isaacus-dev/legal-rag-bench)):

- FAISS via LangChain, one passage = one document, no chunking, k=5: CORRECT in article.
- Retrieval accuracy is deterministic (`relevant_passage_in_context`), no LLM judge: CORRECT.
- Paper judge: GPT-5.2 high reasoning, fixed across all runs: CORRECT.
- 52% retrieval for text-embedding-3-large: CORRECT (our repro matches).
- Kanon 2 Embedder is +34 retrieval points over TE3L (~86%), +17.5 correctness,
  +4.5 groundedness. TE3L is the paper's *second-best* embedder, not the best.
- Paper blog published 2026-02-20; arXiv 2026-03-02.
- Dense run internal consistency: 49 success + 18 abstention + 17 hallucination
  + 16 errors = 100. Matches implementation memory.

## Blocking fixes (before publishing)

| # | Fix | Where | Status |
|---|-----|-------|--------|
| 1 | "best embedding model tops out at 52%" is FALSE (Kanon 2 ~86%). Change to "best general-purpose embedder" + add a sentence acknowledging Kanon 2 (preempts the obvious rebuttal: domain embeddings are a bigger lever than architecture) | long article, intro (line 5) | **done 2026-07-02** |
| 2 | "Last month" → "In February" (blog is 2026-02-20, now July) | long article, intro | **done 2026-07-02** |
| 3 | Verify final agent run's `config_snapshot` really used `deepseek/deepseek-v4-flash` (earlier FS run on 2026-06-27 used `openai/gpt-5.4-mini`). If different, the "same generator" claim is false and must be fixed/disclosed | backend DB, then both articles | **verified OK 2026-07-02** — all 3 final runs used deepseek-v4-flash (dense `06c2f8da…`, hybrid `882e7dd1…`, fs `4d5e5ad0…` in `storage/dev.db`); gpt-5.4-mini runs were the 06-27 subset experiments |
| 4 | Mixed denominators: 75% correct (of 88) vs 59% gold-access (of 100). Add conservative all-100 number: timeouts-as-failures ⇒ 66/100 = 66% correct, still 17 pts above dense | long article Result 3 + LinkedIn item 3 | **done 2026-07-02** (both articles + timeout caveat) |
| 5 | "64 clean successes" ≠ 75%×88=66. Define (taxonomy success = correct ∧ grounded) or use 66 correct | long article, Result 3 | **done 2026-07-02** — reworded as "64 answers both correct *and* grounded" (matches taxonomy success=64 in DB) |
| 6 | "Abstained" column: raw counts (18, 32, 2) with different denominators (100, 97, 88) among % columns → convert to % | long article, headline table | **done 2026-07-02** — 18.0% / 33.0% / 2.3% over judged cases; caption updated |
| 7 | "a handful timed out" (caption) vs "12 of the 100" (cost section) — 12 is not a handful; unify | long article | **done 2026-07-02** |

## High-impact improvements (long article)

- [x] **(a) Bar chart done 2026-07-02**: `docs/images/legal-rag-retrieval-vs-correctness.png`
  (+ `-dark.png` variant), embedded after the headline table via `<picture>`;
  reuse the light PNG as the LinkedIn attachment. Regenerate with
  `scratchpad make_chart.py` (matplotlib) if numbers change.
- [ ] **(b) 1–2 platform screenshots** (comparison view, retrieval trace).
- [ ] **Links.** Repo (currently zero links to it!), paper/arXiv, HF dataset,
  exported comparison files ("manifest included" — link it).
- [ ] **"Reproduce this" section** with the exact commands + commit/tag of the
  code used for the runs.
- [ ] **Quantify cost.** Title says "sent me a bill" but no $ appears. Platform
  tracks tokens/cost — add $/question per architecture to the table.
- [ ] **Trim the platform entity list** (6 bullets are internal vocabulary; keep
  the two ideas: frozen snapshot travels with result, same code everywhere;
  link repo for details).
- [ ] **Author footer.** 2 lines: who you are + GitHub/LinkedIn links (job-search
  artifact — reader must find you in one click).
- [ ] Consider moving "Setup, stated plainly" before the platform section, or
  compress the ~900 words between headline table and Result 1.

## LinkedIn post improvements

- [ ] **Rewrite hook** (first 2–3 lines decide "see more"). Lead with the
  counterintuitive result, e.g.:
  > Hybrid search *lost* to plain dense retrieval on a legal benchmark — 41% vs 52%.
  > And an agent that just reads files beat them both, 26 points above the "retrieval ceiling."
  > Same corpus, same embeddings, same judge. Only the architecture changed.
- [ ] **Attach the bar-chart image** (single biggest reach multiplier).
- [ ] Compress the caveat paragraph to one line.
- [x] Move "top-5" so it applies only to the vector systems (done 2026-07-02).
- [x] Conservative 66/100 number added to item 3 (done 2026-07-02).
- [ ] Remove the `# LinkedIn post` draft header before posting.
- [ ] Decide link placement: in-post (reach penalty) vs first comment.

## Publication strategy (decided + open)

Sequence:

1. Fixed-judge decision (below) → finalize numbers.
2. Merge `legal-rag-bench` branch to main (with ports per port-status report).
3. Add "Case study" link in README top section (article ↔ repo must link both ways).
4. Publish long article on main; LinkedIn post with chart image; pin post +
   add article to profile Featured section.
5. Days later: cross-post to Medium (canonical → GitHub) for discoverability.
6. When repo is polished (quickstart works first try): Show HN for the
   *platform*; Reddit r/Rag, r/LocalLLaMA, r/MachineLearning [P].

Not recommended: native LinkedIn Articles (poor reach).

## Open decision (user's call)

**Publish now (self-judge) vs fixed-judge rerun first.** The article itself
calls the fixed judge "the cleanest single upgrade to credibility" and says
Phase 2 should happen "before quoting these as final". Recommendation: rerun
judging with one fixed strong judge (~300 judgments × 3 runs, modest cost)
before publishing; it converts 3 of 5 caveats into footnotes. If time-to-publish
wins, the disclosure as written is honest — expect "self-judge" as the first
objection in comments.

## Changelog

- 2026-07-02: Initial review completed; plan created.
- 2026-07-02: All 7 blocking fixes resolved (fix 3 verified against `dev.db` —
  "same generator" claim holds). Results bar chart generated (light + dark)
  and embedded in the long article. LinkedIn micro-fixes applied.
