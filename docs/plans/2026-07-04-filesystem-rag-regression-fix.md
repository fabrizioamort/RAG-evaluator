# Filesystem RAG Round-2 Regression Fix Plan

**Date:** 2026-07-04
**Branch:** `legal-rag-bench`
**Trigger:** Comparison `52a891f8` — the 10-question Legal RAG Bench re-eval run after the
round-2 optimizations (eval `8401dfd3`, 2026-07-04) regressed against the pre-optimization
run (eval `bff7bd30`, 2026-07-02). Same RAG config parameters, same judge
(`deepseek-v4-flash` via OpenRouter) in both runs.

## 1. Evidence

| Signal | Old (Jul 2, bff7bd30) | New (Jul 4, 8401dfd3) |
|---|---|---|
| Judge correct | 0.8 | 0.6 |
| Judge grounded | 0.9 | 0.6 |
| Taxonomy success | 0.8 | 0.5 |
| gold_accessed | 0.9 | 0.8 |
| Cost / prompt tokens | $0.019 / 203K (estimated, see below) | $0.306 / 3.3M (provider-reported) |

**Token caveat (verified 2026-07-04):** the two token/cost numbers are NOT directly
comparable. Before `a330da8` the Filesystem RAG *estimated* tokens (`len(prompt) // 4`,
not summed across agent iterations); `a330da8` switched to provider-reported usage
accumulated over every LLM call. The old run's true usage was materially higher than
203K. The *real* efficiency regression is shown by behavior, not the token delta:
Josh ran ~59 tool calls / ~39 of 40 iterations (near the ceiling) in the new run, and
per-question conversation growth is unbounded either way.

**BM25 quality measurement (verified 2026-07-04, script in §4 Phase 0):** scoring each of
the 10 questions against the new index (`idx_46d32975669e4340bc6d031f`) with
`BM25PassageIndex.search(question, top_k=30)`: the gold passage lands in the top 5 for
only **1/10** questions (Bob & Ted, rank 2 — the question that improved). 5/10 are not in
the top 30 at all. A stopword-filtered query variant was also measured: still 1/10 in
top-5 (Frank & Joe got *worse*, rank 6 → >30). Conclusion: the failure is a vocabulary
gap (scenario-style questions vs. statute-book passage wording), and stopword filtering
alone is not a fix — reformulation and union retrieval are.

Question-level flips (from raw metrics artifacts):

- **Emma (circumstantial evidence)** — regressed, gold True→False. Gold passage is
  `3.6-c2-s1` ("Use of Circumstantial Evidence"). The old lexical prefetch surfaced it
  directly (as `doc_125`); the new BM25 prefetch and `search_passages` calls only returned
  chapter-7 passages ("Recent Possession" `7.5.11`, "Distress" `7.3.1.7`). The agent
  satisficed on the sibling — but note the final answer still reached the same correct
  conclusion (G-Eval 1.0 in both runs), reasoned from the recent-possession doctrine
  instead of the gold passage. The regression here is retrieval + judged grounding, not
  answer substance.
- **Frank & Joe (arson)** — regressed, gold True→False. Same crowd-out pattern.
- **Jurors / news stories** — regressed, correct True→False *with gold accessed*. The agent
  anchored on the empanelment model charge that BM25 surfaced first and answered "Yes";
  the old run reasoned from the pre-trial-publicity authorities and answered "No".
- **Josh (evidentiary process)** — regressed, grounded True→False *with gold accessed*.
  The agent thrashed: ~39 iterations, 59 tool calls, 2.15M prompt tokens on one question.
- **Bob & Ted** — improved (hallucination → success).
- **Harry** — failed in both runs (taxonomy category changed only; G-Eval actually
  improved 0.2 → 0.7).

**Judged vs. substantive quality (added after review, 2026-07-04):** G-Eval per question
shows only ONE outright wrong answer in the new run (news stories, G-Eval 0.0). Emma and
Frank & Joe scored G-Eval 1.0 and Josh 0.8 while the legal judge marked them
correct=False and/or grounded=False. G-Eval average moved just 0.91 → 0.85. So the
judged regression decomposes into: (a) a real retrieval regression (gold missed on 2
questions), (b) one real reasoning regression (news stories), and (c) a judged-grounding
drop on substantively correct answers — which interacts with RC3 point 4 (the new judge
caps the context it sees at 40K, and the new run's contexts are far larger). The
alternate-evidence crediting from `264788c` did not catch Emma/Frank & Joe.

## 2. Root causes

### RC1 — BM25-first navigation anchors the agent on near-miss siblings

Commit `a330da8` made BM25 the primary entry point in three reinforcing ways:

1. `agent/prefetch.py` (`build_prefetch_context`) injects the top-3 BM25 candidates into
   the system prompt as `_index/passages/bm25_candidates` with explicit `read_file(...)`
   hints.
2. The system prompt's Navigation Strategy (`agent/prompts.py`, rules 2a–2c) says "use
   search_passages first" for every query type. The topic map (`_index/topics/`) that the
   old run navigated is no longer part of the recommended path.
3. `agent/agent.py::query` seeds `context_chunks`/`context_sources` with the prefetch
   snippets, so unverified BM25 candidates become "retrieved context" for the judge even
   when they are irrelevant — this also dilutes grounding.

When BM25's top hits are plausible siblings from the wrong chapter (model charges vs.
evidence doctrine), the agent commits to them instead of exploring, and nothing in the
prompt forces a cross-chapter sanity check.

Secondary: `passage_index.py::tokenize` does no query-side stopword filtering — the Emma
prefetch searched with terms like "is", "of", "but", "the". BM25 IDF downweights them in
scoring, but they occupy `matched_terms`, inflate weak matches, and add noise to the
candidate ranking.

### RC2 — Unbounded conversation growth (cost + grounding regression)

`agent/agent.py::query` appends every tool result to `messages` and never prunes. With
`TOOL_RESULT_LIMITS` raised in `fea4212` (read_file 10K chars, search 6–8K) and config
caps of 40 iterations / 80 tool calls, the conversation grows quadratically: Josh's
question re-sent an ever-growing history 59 times → 2.15M prompt tokens. (The headline
"16x cost" vs. the old run overstates the regression because the old run's tokens were
estimated, not measured — see §1 token caveat — but the near-ceiling thrash is real.)
The two grounded=False-despite-gold cases (Josh, Harry) are consistent with the model
assembling answers from a huge noisy context instead of the gold evidence — though the
judge's own 40K context cap (RC3 point 4) is a competing explanation; Phase 0 step 2
separates the two.

### RC3 — Uncontrolled experiment

At least four things changed between the two runs, not one:

1. **Prepared index:** old run used `idx_844272ee548a4fe19c05ad52` (built 2026-06-28,
   `doc_NNN.md` + topic map, no BM25); new run used `idx_46d32975669e4340bc6d031f`
   (built 2026-07-03, passage-named files + `_index/passages/bm25.json`).
2. **Agent code:** `a330da8` (BM25-first) + `fea4212` (budgets) + round-2 fixes.
3. **Token accounting:** estimate → provider-reported (see §1 caveat).
4. **Judge/metric code:** `4050d68` ("Harden Legal RAG judge", 2026-07-03) and `264788c`
   (split success signals, credit alternate evidence) landed between the runs, and
   `a3e2ff5` (2026-07-04 13:37) landed 90 minutes before the new eval. `4050d68` is NOT
   just parse-robustness: it also **bounds the judge's visible context (40K total, 8K per
   chunk) and excludes navigation-index chunks**. The old run was judged with unbounded
   context; the new run — whose contexts are far larger — was judged through the cap.
   This plausibly contributes to the grounded=False-despite-gold cases (Josh, Harry) and
   the Emma/Frank & Joe correct=False verdicts on substantively correct answers (§1).
   The alternate-evidence crediting in `264788c` cuts the other way (more lenient), so
   the direction of the net judge-side effect is unknown — measure it (Phase 0 step 2).

n=10 means one question = 10 points. Any further change must be validated on a fixed
corpus index and, before final conclusions, on the full 28-case set.

## 3. Guiding constraints

- One phase per evaluation run. Never change retrieval behavior and budgets in the same
  run — we just paid for that mistake.
- Keep the judge fixed (`deepseek-v4-flash` for A/B continuity; the article's fixed
  GPT-5.2 judge run happens after behavior is stable).
- Keep the prepared corpus index fixed for all runs in this plan (the current
  passage-named layout). Do not re-prepare.
- Do not lower `TOOL_RESULT_LIMITS` back down as a first resort — `fea4212` fixed a real
  problem (the agent could not see the evidence it read). Fix history growth instead.

## 4. Phases

### Phase 0 — Lock the baseline and build a retrieval-only harness

1. **Retrieval regression harness (no LLM calls):** add
   `scripts/legal_rag_retrieval_check.py` that, for a locked set of question/gold pairs,
   loads the prepared index's `BM25PassageIndex` (and later, the full prefetch pipeline)
   and reports the rank of the gold passage. Seed it with the 10 subset questions
   (gold = `relevant_passage_id` from each run's raw metrics artifact); the exact
   failures to track by name: Emma (`3.6-c2-s1`), Frank & Joe (`1.5-c8-s1`),
   juror news stories (`1.5-c7-s2`), Josh (`2.3.3-c2-s1`), Harry (`1.5-c5-s1`).
   Baseline measured 2026-07-04: 1/10 gold in top-5, 5/10 not in top-30.
   This makes every Phase 2 retrieval change validatable in seconds for ~$0.
2. **Quantify the judge-side share of the regression:** re-judge the new run's four
   flipped answers (Emma, Frank & Joe, Josh, Harry — stored in `evaluation_results`)
   with the judge context cap lifted (or pre-`4050d68` judge behavior), same judge
   model. If correct/grounded flip back for the G-Eval-1.0 answers, a chunk of the
   "regression" is a judging artifact of capped context over bloated runs — which
   raises the priority of Phase 1 (smaller contexts fix the judged metric too) and may
   justify a judge-side fix (rank evidence chunks before capping instead of truncating
   in order).
3. Re-run the 10-question subset once more on current HEAD, same config, same judge, to
   confirm `8401dfd3` is reproducible and not a one-off (agent runs are stochastic;
   3 flipped questions could partly be variance).
4. Record the run ids next to `8401dfd3` in the memory file
   `filesystem-rag-postfix-failure-analysis.md`.

**Exit criterion:** harness runs and reproduces the 1/10 baseline; we know whether the
end-to-end regression reproduces (expected: mostly yes for the two gold misses, possibly
variance on the reasoning flip).

### Phase 1 — Stop the context bloat (RC2)

Target: prompt tokens per question back within ~2x of the old run without losing the
raised per-result visibility.

1. **History compaction** in `agent/agent.py::query`:
   - After a tool result has been consumed (i.e. it is more than N=2 assistant turns
     old), replace its `content` in `messages` with a stub:
     `"[result elided — {tool_name}({key args}); re-call the tool if needed]"`.
     Keep the most recent 2 iterations' results at full size.
   - Exempt `read_file` results whose path the agent has flagged as evidence? No —
     evidence is already preserved separately in `context_chunks`
     (`_context_chunk_from_tool_result`), which is what the final answer and judge see.
     Elide uniformly; the stub tells the model how to get it back.
   - Implementation detail: track `(message_index, iteration, tool_name, args)` for each
     appended tool message; compact at the top of each loop iteration.
2. **Wrap-up nudge:** when `iteration >= max_iterations * 0.6` or
   `tool_call_count >= max_tool_calls * 0.6`, append a one-time system nudge: "You have
   used most of your budget. Synthesize the best-supported answer from evidence already
   gathered unless one specific missing fact blocks you." (The existing
   `evidence_nudge_used` / `_synthesize_partial_answer` machinery in `agent.py` is the
   pattern to follow; this is a new, earlier nudge.)
3. **Duplicate-call guard:** keep a set of `(tool_name, canonical_args)`; on a repeat
   call return a cached short reference to the earlier result instead of re-emitting the
   full payload. Josh's 59 calls almost certainly contain re-reads.

**Validation:** re-run the 10-question subset. Acceptance criteria are absolute (the old
run's 203K is an estimate and not a valid target, per §1 token caveat): no question over
150K provider-reported prompt tokens; no question within 90% of the iteration or
tool-call ceiling; judge metrics not worse than Phase 0 baseline.

### Phase 2 — Fix BM25 anchoring (RC1)

Target: recover gold access on Emma / Frank & Joe without losing the Bob & Ted win.
Ordered by measured leverage; validate each item with the Phase 0 harness before the
next, and stop when the harness shows gold in top-8 for ≥ 8/10 questions.

1. **Union prefetch with a reformulated query** (highest leverage — the measured failure
   is a vocabulary gap, see §1). `build_prefetch_context` currently runs one BM25 search
   with the raw scenario-style question. Add candidates from:
   - BM25 on the raw question (current behavior);
   - BM25 on a doctrinal reformulation (the router already extracts legal-issue terms;
     the agent prompt's "two vocabularies" rule, applied at prefetch time);
   - the question-seed index (`_index/questions/`) as navigation hints only.
   Dedupe, merge by score, surface the top ~8 instead of 3.
2. **Section-family diversity in prefetch candidates:** when merging, cap candidates per
   section family (e.g. max 2 sharing a `1.2*` prefix) so one doctrine family cannot
   fill the whole list — Emma's top hits were all chapter 7, news-stories' all
   empanelment. Implement in `prefetch.py` at merge time, not inside `BM25PassageIndex`.
3. **Demote prefetch from evidence to hypothesis:**
   - In `agent/agent.py::query`, stop seeding `context_chunks`/`context_sources` from
     `prefetch["chunks"]`. Prefetch snippets stay in the system prompt as navigation
     hints only; they enter evidence only when the agent actually reads the file.
   - Reword the prefetch block header: "These are keyword-match candidates, not verified
     answers. Treat them as one hypothesis; check the topic index before committing."
4. **Restore the topic map as a co-equal entry path** in `agent/prompts.py` Navigation
   Strategy: replace the "search_passages first" rules with: (a) run `search_passages`;
   (b) in parallel, check `_index/topics/` for the doctrine the question is about;
   (c) **chapter-mismatch rule:** "If all top BM25 hits come from a model-charge or
   bench-notes chapter but the question asks about admissibility, evidence doctrine, or
   procedure, the answer likely lives in a different chapter — search the topic index
   and re-query with the doctrinal term before reading siblings." (This is the exact
   Emma/news-stories failure, stated as an instruction.)
5. **Contrary-evidence check for yes/no questions:** add a Navigation Strategy rule:
   before finalizing a yes/no answer, run one search phrased for the opposite conclusion
   and read the best contrary candidate if it lands in a nearby doctrine family. This
   targets the news-stories failure mode (gold accessed, answer anchored on the first
   plausible passage). Cheap: one rule + at most 1–2 extra tool calls per question.
6. ~~Query-side stopword filtering~~ — **measured, rejected as a primary fix**: filtered
   queries still put gold in top-5 for only 1/10 and made Frank & Joe worse (§1). May be
   revisited as hygiene inside item 1's reformulated query, but never as its own phase.

**Validation:** harness after each item; then re-run the 10-question subset once the
harness gate (gold in top-8 for ≥ 8/10) passes. Acceptance: gold_accessed ≥ 0.9 (Emma
and Frank & Joe recovered), Bob & Ted still success, judge correct ≥ 0.8. If Emma still
misses, inspect her trace before adding anything else — do not stack more heuristics
blind.

### Phase 3 — Grounded final synthesis (residual grounding gap)

Only if grounded < 0.9 after Phases 1–2 (Josh/Harry pattern persists):

1. When the agent emits its final answer, run one additional LLM call that rewrites the
   answer using **only** the accumulated `context_chunks` (the bounded evidence channel),
   with the existing Answer Contract rules. This decouples answer wording from the noisy
   navigation history entirely.
2. Cost note: one extra call per question is ~10–30K tokens with compacted evidence —
   trivial next to the current overspend.

### Phase 4 — Full validation and article data

1. Run the full 28-case set with the final code, same judge, same corpus index.
   Targets (vs. the 2026-07-03 28-case run: gold 14/28): gold_accessed ≥ 0.75,
   judge correct ≥ 0.65, grounded ≥ 0.85, cost per 28 questions ≤ $0.15.
2. Only after 28-case targets are met: the fixed-judge (GPT-5.2) rerun for the article
   comparison tables, per `legal-rag-closed-vs-open-book` memory.
3. Update `docs/articles/legal-rag-bench-architecture-comparison.md` numbers and the
   memory files; commit per phase with the eval ids in the commit message.

## 5. Non-goals / explicitly rejected

- **Reverting `fea4212` (tool-result budgets).** The pre-raise failure mode (agent blind
  to evidence it read) was real; RC2 is history accumulation, not per-result size.
- **Reverting the BM25 index (`a330da8`) wholesale.** It fixed Bob & Ted and gives the
  agent a real search primitive; the problem is exclusive reliance, not existence.
- **Re-preparing the corpus.** Layout changes invalidate A/B comparison again (RC3).
- **Raising iteration/tool caps.** Josh shows more budget produces thrash, not answers.

## 6. Risks

- **n=10 variance:** each subset run can flip ±1 question for free. Phase gates use
  direction + trace inspection, not single-metric thresholds alone. The 28-case run is
  the real gate.
- **History compaction breaking tool-call protocols:** some providers require tool
  messages to remain paired with their assistant tool_calls message. Compaction replaces
  content, never removes messages — verify with a single smoke question against the
  configured provider before the eval run.
- **Prompt-rule inflation:** the Navigation Strategy is already 10 rules. Phase 2 should
  net-replace rules 2a–2c, not append; if the prompt grows past ~15 rules, consolidate.
