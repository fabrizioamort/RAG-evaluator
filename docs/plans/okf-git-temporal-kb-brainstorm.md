# Git-Enhanced OKF for Evolving Knowledge Bases

**Date:** 2026-07-12  
**Status:** Design brainstorm and research notes  
**Related proposal:** `docs/plans/2026-07-09-okf-rag-implementation-plan.md`

## Executive conclusion

Git could be extremely valuable for an evolving Open Knowledge Format knowledge base. However:

> Git is an excellent version ledger for an OKF bundle, but it is not, by itself, a temporal knowledge model.

The strongest design combines Git history with explicit validity metadata inside the knowledge representation.

Git can reliably answer questions such as:

- What did the knowledge base contain at a particular revision?
- Which files changed during an ingest?
- Who or which agent proposed and accepted the change?
- Can a problematic enrichment be rolled back?
- Can an evaluation be reproduced against exactly the same bundle?

Git cannot answer by itself:

- When was an offer actually valid in the business world?
- Was a correction retroactive or prospective?
- Is a previous claim expired, withdrawn, incorrect, or still valid for a subset of customers?
- Which version should be used for a query about a historical date?

Those require an explicit temporal model layered on top of OKF.

## What OKF already says about evolution

OKF acknowledges change, but only lightly.

The v0.1 draft includes:

- an optional `timestamp` on concepts;
- an optional `log.md` for chronological updates;
- Git as the recommended distribution mechanism.

The specification says that a bundle may be distributed as a Git repository, but it does not define supersession, effective dates, expiration, point-in-time queries, branching, or how a consumer decides which conflicting fact is current. See the [OKF v0.1 specification](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md).

Karpathy goes slightly further: the original LLM Wiki proposal describes the wiki as a Git repository, noting that version history, branching, and collaboration come naturally. It still does not define temporal retrieval semantics. See [Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

Git is therefore already philosophically compatible with both ideas. What is missing is the temporal knowledge layer above Git.

## What Git would give OKF

Git is particularly well matched to Markdown knowledge because it provides:

- immutable snapshots of the complete bundle;
- meaningful line-by-line diffs;
- rollback after a bad LLM enrichment;
- attribution of human and agent changes;
- branches for proposed knowledge updates;
- pull-request review before publication;
- tags for published KB releases;
- reproducible evaluation against an exact commit;
- change detection for incremental rebuilding;
- an audit trail explaining when the system learned something;
- comparison of the knowledge graph before and after an ingest.

That last capability is especially interesting for RAG Evaluator. The platform could evaluate:

```text
Question Q against KB commit A
Question Q against KB commit B
Question Q against KB commit C
```

This would show whether an evolving KB actually improves, regresses, or develops stale-fact errors.

There are already practical LLM-wiki implementations using this pattern. For example, [MehmetGoekce/llm-wiki](https://github.com/MehmetGoekce/llm-wiki) runs an ingest pipeline, quality-checks the resulting page changes, and then commits the transaction to Git. Its query workflow can also propose new pages when it discovers a knowledge gap.

No official OKF proposal was found that defines Git-based temporal semantics. The official OKF repository uses and recommends Git, but currently treats it as packaging, collaboration, and history rather than as a formal validity model.

## The critical distinction: three kinds of time

Suppose an offer is valid from September through November, its terms change in October, and the marketing team does not ingest the correction until a week later.

There are at least three different timestamps:

| Time | Meaning |
|---|---|
| Valid time | When the offer or terms are true in the business world |
| Source or event time | When the source document was published or changed |
| Knowledge or transaction time | When the KB ingested and accepted the information |

Git primarily records the third one: when the repository accepted a state.

It does not tell us when an offer became legally or commercially valid.

A Git commit made on October 10 might contain:

- an offer effective since October 1;
- a future offer effective November 1;
- a correction saying that something believed since September was never valid;
- a late-ingested document published in August.

Git author and commit dates are also configurable, so they are not authoritative business timestamps. Git distinguishes author and committer metadata and calculates changes by comparing commit trees; it does not intrinsically understand the semantic meaning of a change. See the [Git user manual](https://git-scm.com/docs/user-manual).

Therefore:

```text
Git commit time != source publication time != factual validity time
```

This is the most important design principle.

## A concrete marketing-offer example

Imagine these events:

```text
August 15:
  Offer A is ingested.
  Valid from September 1 through November 30.

October 3:
  New terms are published.
  They become valid on October 10.

October 7:
  The KB ingests the new document.

November 20:
  Offer B is announced.
  It replaces Offer A on December 1.
```

Git can answer:

- What did the KB contain on October 5?
- Which files changed when the new terms were ingested?
- Who or which agent accepted the update?
- Can the KB state from before the update be restored?
- Did answer quality improve after that commit?

Git cannot answer by itself:

- Which terms were valid on October 12?
- Was the October update retroactive?
- Which offer should be shown to a customer today?
- Was an old claim corrected, superseded, or merely removed?
- Did the KB learn the change after it became effective?

Those require explicit temporal metadata.

## A Git-enhanced OKF profile

This should be defined as a RAG Evaluator extension rather than as a change to core OKF.

For example:

```yaml
---
type: Offer
title: Autumn Enterprise Offer
description: Discounted enterprise package for the autumn campaign.
timestamp: 2026-10-07T14:30:00Z
tags: [enterprise, campaign, autumn]
rag_evaluator:
  stable_id: "offer:autumn-enterprise-2026"
  lifecycle: active
  valid_from: 2026-10-10T00:00:00Z
  valid_until: 2026-11-30T23:59:59Z
  supersedes:
    - "/offers/history/autumn-enterprise-v1.md"
  source_revision_ids:
    - "sha256:..."
---
```

The fields have different purposes:

- `timestamp`: OKF's last meaningful document change.
- `valid_from` and `valid_until`: business validity.
- `lifecycle`: draft, scheduled, active, expired, superseded, or withdrawn.
- `stable_id`: identity across renames and file moves.
- `supersedes`: explicit semantic relationship.
- `source_revision_ids`: exact source evidence behind this state.

Do not place the current Git commit SHA inside the committed file itself: a commit cannot contain its own final hash. The runtime index or platform database should associate the bundle fingerprint with the resulting commit SHA after the commit is created.

## Should old knowledge live only in Git?

There are two main designs.

### Option A: current state in OKF, history only in Git

The canonical offer page is updated in place. Git retains older versions.

Advantages:

- very simple current bundle;
- fewer files;
- ordinary OKF consumers see only current knowledge;
- Git diffs are straightforward.

Disadvantages:

- a non-Git export loses semantic history;
- historical queries require checking out or reading old commits;
- relationships to old versions are difficult;
- Git diffs operate at file and line level, not claim level.

### Option B: current and historical versions are explicit OKF concepts

The bundle contains a canonical current page and immutable version pages:

```text
offers/
├── autumn-enterprise.md
└── history/
    ├── autumn-enterprise-v1.md
    └── autumn-enterprise-v2.md
```

Advantages:

- temporal information survives ZIP or tar export;
- any OKF consumer can inspect prior terms;
- current pages can link to what they supersede;
- historical retrieval does not require Git.

Disadvantages:

- more files and duplication;
- current retrieval must filter expired versions;
- the producer must maintain consistency between canonical and historical pages.

### Recommended hybrid

Use:

- one canonical concept representing the current materialized view;
- immutable source revisions and significant historical concept versions;
- explicit validity and supersession metadata;
- Git for complete transaction history and review.

This avoids forcing ordinary queries to inspect every historical revision while preserving important temporal states in portable OKF.

Git is then the audit ledger; OKF metadata is the semantic time model.

## A good Git workflow

Do not automatically commit after every tool call or query. Use one commit per accepted knowledge transaction.

```text
new or changed source
        ↓
create ingest branch
        ↓
extract source revision
        ↓
update affected OKF concepts
        ↓
compute semantic diff
        ↓
validate links, evidence, validity intervals
        ↓
human or policy approval
        ↓
merge into published branch
        ↓
build runtime index bound to commit SHA
```

Suggested branch roles:

- `main`: published, queryable KB.
- `ingest/<source-id>`: proposed source ingestion.
- `maintenance/<issue>`: link repairs, deduplication, and reorganization.
- `experiment/<name>`: alternative LLM enrichment or schema experiments.

Useful tags:

```text
kb/2026-09-01
kb/2026-10-terms-update
kb/2026-q4-release
```

Commit messages could include machine-readable trailers:

```text
Update Autumn Enterprise Offer terms

OKF-Change-Type: source-update
OKF-Source-Revision: sha256:...
OKF-Effective-From: 2026-10-10T00:00:00Z
OKF-Producer-Model: ...
OKF-Prompt-Digest: sha256:...
OKF-Reviewed-By: user:123
```

This makes history queryable without overloading frontmatter with build-operation details.

## Four temporal query modes

A Git-aware OKF RAG should distinguish which temporal question the user is asking.

### 1. Current truth

> What enterprise offer is active today?

Query `main` and filter concepts or claims by their validity interval.

### 2. Valid-at-time

> What offer was valid on October 12?

Use `valid_from` and `valid_until`, not Git commit dates.

### 3. Known-at-time

> What did our sales KB believe on October 5?

Query the repository commit that was published as of October 5.

### 4. Change or audit

> How did the offer change, when was it changed, and why?

Compare concept versions, sources, validity metadata, and Git commits.

These questions sound similar but can return different answers. Making that distinction explicit would be a genuinely original and scientifically interesting extension to OKF RAG.

## Why temporal metadata matters for retrieval

Recent work supports the concern that ordinary RAG behaves poorly when facts evolve. A recent temporal-validity study reports that embeddings often retrieve stale and current facts with nearly identical similarity because contradictions use much of the same vocabulary. It proposes retiring superseded facts in a temporal ledger rather than expecting semantic similarity to solve the problem. This is recent preprint work, so its exact quantitative findings should be treated cautiously, but the failure mode is convincing. See [Temporal Validity in Retrieval Memory](https://arxiv.org/abs/2606.26511).

The adjacent [Graphiti project](https://github.com/getzep/graphiti) follows a similar principle: it gives facts validity windows, preserves superseded facts, and maintains provenance back to raw episodes. It is a useful architectural reference even though it is a temporal graph system rather than OKF.

A Git-enhanced OKF could provide many of the same properties using transparent Markdown rather than requiring a graph database.

## What could go wrong

### Git history is not semantic history

Deleting one sentence and adding another shows that text changed. It does not explain whether the old statement:

- expired naturally;
- was wrong;
- was superseded;
- was narrowed by an exception;
- was moved elsewhere;
- remains valid for some customers.

That relationship requires explicit metadata or a claim/event ledger.

### `git blame` is weaker than it looks

Blame identifies the commit associated with a current line. It is not necessarily the original source, responsible business owner, or effective date. Reformatting or LLM rewrites can also destroy useful blame continuity.

Use Git provenance for “who changed this representation,” and source citations for “who established this fact.”

### Git does not explicitly record renames

Git infers renames by comparing content. Stable concept identity should therefore not depend only on the current file path. The Git documentation notes that renames are detected from similar content rather than stored as intrinsic rename operations. See the [Git user manual](https://git-scm.com/docs/user-manual).

### Automated commits can produce useless history

If every enrichment rewrites entire files, Git history becomes noise. Deterministic serialization and surgical updates are prerequisites.

One accepted ingest should ideally produce one coherent commit whose diff answers:

> What did this source cause the knowledge base to learn, revise, expire, or retract?

### Multi-agent writes create semantic merge conflicts

Git resolves text, not knowledge. Two agents may modify different lines of the same concept without a textual conflict while creating a semantic contradiction.

Use branches and serialized publication, then run semantic validation after merging.

### Removed information remains in history

This is beneficial for audit but dangerous for:

- credentials;
- personal information;
- confidential pricing;
- legally required deletion;
- licensed source material.

Deleting a file from the current branch does not remove it from Git history. History rewriting can remove it, but changes commit identities and invalidates signatures and stored references. A retention and redaction policy is necessary before committing sensitive KBs.

### Repository growth

Markdown diffs compress well, but extracted PDFs, images, generated indexes, and frequently rewritten large files can make the repository enormous.

Commit:

- portable Markdown knowledge;
- small manifests and schemas;
- possibly normalized source text.

Do not commit:

- runtime search indexes;
- embeddings;
- caches;
- temporary reports;
- large source binaries by default.

Keep binaries in the platform artifact store and reference them by immutable hash. Git LFS is possible, but introduces another service and weakens the “just clone it” property.

### Commit identity is not authorization

Git author metadata can be set by the client. Signed commits and protected branches improve confidence, but enterprise audit may still require server-side acceptance timestamps and reviewer identities.

## Relationship to RAG Evaluator

The platform already has a `KnowledgeBaseVersion` model containing:

- a monotonically increasing version number;
- a change type;
- a document snapshot;
- a change description.

Evaluations and indexes can refer to a KB version.

That is useful, but it is currently a coarse document-set snapshot. It does not model:

- OKF concept-level diffs;
- business validity;
- supersession;
- Git commit identity;
- historical bundle checkout.

If Git is introduced, avoid two competing version histories. A future integration should map a platform KB version to an accepted OKF commit or tag and bundle fingerprint. The platform database remains authoritative for access control and evaluation linkage; Git provides artifact history and diffs.

## What others have already done

The research found this landscape:

- The official OKF draft recommends Git repositories for bundle distribution but defines no Git-aware temporal query semantics. See the [OKF specification](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md).
- Karpathy explicitly says the wiki can be a Git repository and gain history, branching, and collaboration. See the [original gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).
- [MehmetGoekce/llm-wiki](https://github.com/MehmetGoekce/llm-wiki) makes Git commit the final step of each accepted ingest pipeline.
- Other community implementations use Git for visible agent history, backup, or repository-native operation, but no mature implementation was found that combines OKF, Git snapshots, explicit valid-time intervals, and `known-at` versus `valid-at` query semantics.
- Temporal graph projects such as [Graphiti](https://github.com/getzep/graphiti) demonstrate that validity windows, supersession, and provenance are important for evolving agent knowledge, but use databases rather than Git-backed OKF Markdown.

The broad “Git-backed LLM wiki” idea already exists. The more specific idea—

> Git-versioned OKF with bitemporal retrieval and reproducible RAG evaluation

—still appears meaningfully original based on the sources found.

## A sensible research roadmap

### Experiment 1: Git snapshots only

- Commit each successfully published OKF bundle.
- Bind every runtime index and evaluation to a commit SHA.
- Support diff, rollback, and query against a selected commit.
- Do not change retrieval ranking yet.

This establishes reproducibility and audit value.

### Experiment 2: explicit validity

Add:

- `stable_id`;
- `valid_from`;
- `valid_until`;
- `lifecycle`;
- `supersedes`;
- immutable source revisions.

Default retrieval excludes expired and superseded knowledge.

### Experiment 3: temporal queries

Classify queries into:

- current;
- valid-at;
- known-at;
- change/audit.

Evaluate classification errors separately from retrieval errors.

### Experiment 4: controlled maintenance

Use branches for agent proposals, semantic diffs, validation, and human acceptance. Never let ordinary evaluation queries mutate `main`.

### Experiment 5: compounding evaluation

Feed a chronological sequence of source changes and measure:

- stale-answer rate;
- correct supersession;
- temporal retrieval accuracy;
- cost of incremental maintenance;
- concept and link churn;
- regression on unchanged knowledge;
- ability to reproduce an old answer at its original commit;
- difference between valid-at and known-at answers.

## Recommendation

Add Git, but not as merely “run `git init` after building.”

Define a first-class, optional **Git history backend** for OKF bundles with these rules:

- one commit per accepted ingest or maintenance transaction;
- no commits during ordinary queries;
- `main` is the published KB;
- agent changes arrive through branches or staging;
- runtime indexes are bound to a commit SHA and kept outside Git;
- business validity lives in explicit OKF-profile metadata;
- source revisions are immutable and content-addressed;
- historical retrieval distinguishes valid-at from known-at;
- the platform database maps its KB versions to Git commits;
- sensitive-data and history-retention policies are mandatory.

That is substantially more powerful than automatic version control. It turns Git into the transaction and audit layer of an evolving, temporally aware knowledge compiler.
