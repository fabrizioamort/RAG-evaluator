# An OKF-Native Temporal Knowledge Repository

**Date:** 2026-07-12  
**Status:** Design brainstorm and research notes  
**Related documents:**

- `docs/plans/2026-07-09-okf-rag-implementation-plan.md`
- `docs/plans/okf-git-temporal-kb-brainstorm.md`

## Executive conclusion

The idea of a new system for managing evolving OKF knowledge makes sense, but it should not initially be framed as a ground-up replacement for Git.

> A new OKF-specific temporal knowledge system could be valuable. A new low-level version-control engine would probably be an expensive distraction.

The missing innovation is not content-addressed storage, branches, or snapshots. Existing systems already solve those. The missing capability is a semantic model for how knowledge becomes valid, is corrected, expires, conflicts, and remains connected to evidence.

A better framing is:

> A temporal, evidence-aware knowledge repository that materializes OKF bundles and borrows the best ideas from version control.

## Existing systems already cover part of the idea

The general idea of applying Git semantics to structured knowledge is established:

- [TerminusDB](https://terminusdb.org/docs/knowledge-graph-version-control/) provides Git-like version control for a knowledge graph, including commits, branches, merges, structured diffs, rollback, cloning, and time-travel over RDF triples.
- [Dolt](https://github.com/dolthub/dolt) provides Git-like version control for relational data, including branches, cell-level merges, history tables, push, and pull.
- Event-sourced architectures store immutable domain events and generate current queryable state as a materialized view. [Microsoft's Event Sourcing pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/event-sourcing) describes this separation.
- Temporal graph systems such as [Graphiti](https://github.com/getzep/graphiti) model facts with validity windows and preserve the source episodes from which they were derived.

The general concept is therefore valid but not entirely novel.

The potentially original contribution is narrower:

> A temporal repository for evidence-backed, partly unstructured knowledge that materializes portable OKF bundles.

TerminusDB works at triple level. Dolt works at table and cell level. Git works at file and blob level. None naturally models an OKF concept containing prose, citations, uncertain synthesis, source excerpts, and changing business validity.

## When Git is sufficient

Git is already enough if the requirements are limited to:

- historical snapshots;
- textual diffs;
- rollback;
- attribution;
- branches and review;
- reproducible evaluation against a revision.

Creating a new system for only these capabilities would be unnecessary.

## When a new system becomes justified

A dedicated system becomes justified if it must understand operations such as:

```text
This offer becomes valid next month.
This claim was corrected retroactively.
This source supersedes another source.
This concept was split into two concepts.
This old policy remains valid for existing customers only.
This claim is disputed by two authoritative sources.
Show what was valid then, not merely what the KB contained then.
```

Git sees these as changed lines. A temporal knowledge system should see them as domain events with explicit meaning.

That is a genuinely different problem.

## The right mental model: a knowledge ledger

The system can be understood as an append-only knowledge ledger that produces OKF snapshots.

```text
sources and human/agent proposals
              ↓
       knowledge changesets
              ↓
    append-only temporal ledger
              ↓
   validation and conflict policy
              ↓
     materialized OKF bundle
              ↓
 search/graph/runtime indexes
```

The OKF directory plays a role similar to Git's working tree: it is a readable projection of a deeper repository state.

The ledger—not mutable Markdown—preserves the complete evolution.

A ZIP export can still contain an ordinary conformant OKF snapshot. Consumers do not need the temporal system unless they want history or advanced temporal queries.

## Core object model

A first design could contain six object types.

### 1. Source revision

An immutable version of an input document:

```yaml
source_id: marketing/offers/autumn
revision_id: sha256:...
observed_at: 2026-10-07T14:30:00Z
published_at: 2026-10-03T09:00:00Z
content_hash: sha256:...
```

Changing a source creates a new revision; it never silently overwrites the old revision.

### 2. Stable concept

Identity independent of title and path:

```yaml
concept_id: offer:autumn-enterprise
current_path: offers/autumn-enterprise.md
```

Renaming or moving its Markdown page does not change its identity.

### 3. Assertion or claim

A knowledge statement with evidence and temporal validity:

```yaml
claim_id: claim:8f193...
concept_id: offer:autumn-enterprise
text: Enterprise customers receive a 20% discount.
valid_from: 2026-10-10T00:00:00Z
valid_until: 2026-11-30T23:59:59Z
evidence:
  - source_revision: sha256:...
    segment: page-3-clause-2
status: accepted
```

This is where the system becomes more than file versioning.

### 4. Relationship

A versioned relationship with meaning:

```yaml
relationship_id: relation:...
from: offer:autumn-enterprise
type: supersedes
to: offer:autumn-enterprise-v1
valid_from: 2026-10-10T00:00:00Z
```

Core OKF links remain untyped Markdown links. Typed temporal relationships belong to the repository or producer-profile layer.

### 5. Knowledge event

An immutable record of what happened:

```yaml
event_type: ClaimSuperseded
claim_id: claim:old-price
replacement_claim_id: claim:new-price
recorded_at: 2026-10-07T14:35:00Z
effective_at: 2026-10-10T00:00:00Z
reason: Updated campaign terms
source_revision: sha256:...
```

Possible event types include:

- `SourceObserved`
- `SourceRevised`
- `ConceptCreated`
- `ConceptRenamed`
- `ConceptMerged`
- `ConceptSplit`
- `ClaimAsserted`
- `ClaimCorrected`
- `ClaimSuperseded`
- `ClaimExpired`
- `ClaimRetracted`
- `RelationshipAdded`
- `EvidenceAttached`
- `ConflictDetected`
- `ChangeAccepted`
- `ChangeRejected`

### 6. Changeset

An atomic proposed update containing multiple related events:

```yaml
changeset_id: change:2026-10-07:autumn-offer
producer: agent:okf-enricher
reviewer: user:123
prompt_digest: sha256:...
status: accepted
events:
  - event:...
  - event:...
```

A source ingest might update ten concept pages. Those changes should be accepted or rejected as one coherent transaction.

## Bitemporal knowledge should be fundamental

The system needs at least two temporal axes:

```text
valid time:    when the knowledge is true in the world
recorded time: when the repository accepted that knowledge
```

For example:

```text
Offer terms valid from:       October 10
Source published:             October 3
Repository learns the change: October 7
```

This permits two distinct historical questions.

“What offer was valid on October 12?”

```text
valid_at = October 12
recorded_at = latest knowledge
```

“What did the company KB believe on October 5?”

```text
recorded_at = October 5
```

The second query may return an older answer even if the system now knows that answer was wrong.

This is bitemporal modeling. It is more expressive than Git time travel because Git only reconstructs what was recorded in a repository state.

## OKF should be a projection, not necessarily the entire database

This is the hardest architectural decision.

### Markdown as canonical state

Advantages:

- simple;
- transparent;
- directly editable;
- no hidden database;
- strongly aligned with OKF.

Problems:

- semantic identity is difficult;
- claim-level changes are hard to detect;
- temporal queries require repeatedly parsing history;
- arbitrary prose edits are difficult to interpret;
- merges remain mostly textual.

### Ledger as canonical state

Advantages:

- semantic events are explicit;
- temporal queries are natural;
- provenance and supersession can be enforced;
- current and historical OKF bundles can be regenerated;
- semantic diffs become possible.

Problems:

- more complex;
- the database can become a second proprietary source of truth;
- editing Markdown requires compiling edits back into events;
- the system risks losing OKF's appealing simplicity.

### Recommendation

Make the ledger canonical for history and accepted semantic changes, while retaining an editable OKF working view.

This resembles Git:

```text
Git object database ↔ working tree
knowledge ledger    ↔ OKF working bundle
```

A user can edit the OKF Markdown. Before accepting it, the system translates the diff into a proposed semantic changeset:

```text
Detected changes:
- Claim C17 changed discount from 15% to 20%.
- Validity begins October 10.
- Source revision S9 was added.
- Existing claim C11 will be superseded.
```

The user or validation policy then accepts or rejects that interpretation.

That translation step is difficult, but it is also the most distinctive and useful capability.

## Do not force all prose into RDF triples

TerminusDB demonstrates that Git-like knowledge-graph versioning is feasible. Copying that design exactly would, however, turn OKF RAG into another graph database.

OKF deliberately supports free-form prose, tables, examples, schemas, and nuanced explanation. Not every paragraph should be decomposed into subject-predicate-object triples.

A tiered model is safer:

- concept-level versioning for every page;
- stable section or block identifiers where possible;
- explicit temporal claims only for facts requiring temporal reasoning;
- opaque Markdown retained for descriptive or explanatory material.

This preserves OKF's readability without pretending that every sentence has a clean ontology.

## Semantic diff is the central feature

A knowledge diff should not primarily say:

```diff
- Customers receive a 15% discount.
+ Customers receive a 20% discount.
```

It should say:

```text
Concept: Autumn Enterprise Offer

Claim superseded:
  Enterprise discount = 15%
  Valid until: 2026-10-09

Claim asserted:
  Enterprise discount = 20%
  Valid from: 2026-10-10

Evidence added:
  campaign-terms revision sha256:...

Affected concepts:
  Autumn Enterprise Offer
  Sales FAQ
  Discount Approval Policy

Potential conflict:
  Sales FAQ still states 15%.
```

This is where a new knowledge-native system could outperform Git substantially.

## Semantic merge is harder than textual merge

A source-code merge generally asks whether lines can coexist. A knowledge merge asks whether beliefs can coexist.

Two branches may say:

```text
Branch A:
  Discount is 20% beginning October 10.

Branch B:
  Discount is 25% beginning October 15.
```

This is not necessarily a direct conflict. It may represent:

- consecutive validity intervals;
- different customer segments;
- one erroneous source;
- one future update;
- two genuinely contradictory proposals.

A semantic merge requires evidence, time, scope, source authority, and possibly human review. It should not automatically choose the latest write.

A useful conflict result would be:

```yaml
conflict:
  kind: overlapping_validity
  subject: offer:autumn-enterprise
  property: discount
  candidates:
    - value: 20%
      valid_from: 2026-10-10
      source: ...
    - value: 25%
      valid_from: 2026-10-15
      source: ...
  resolution_required: true
```

Automatic last-writer-wins is usually wrong for knowledge.

## Possible commands

A prototype could expose a Git-like but knowledge-specific CLI:

```text
okfv init
okfv import bundle/
okfv status
okfv ingest campaign-terms.pdf
okfv propose
okfv diff --semantic
okfv accept <changeset>
okfv reject <changeset>
okfv history offer:autumn-enterprise
okfv explain claim:8f193
okfv expire <claim> --effective 2026-12-01
okfv supersede <old-claim> --with <new-claim>
okfv snapshot --valid-at 2026-10-12
okfv snapshot --known-at 2026-10-05
okfv query --valid-at 2026-10-12
okfv lint-temporal
okfv conflicts
okfv export --format okf
```

The vocabulary changes. Instead of only add, delete, and commit, the knowledge domain has:

- observe;
- assert;
- correct;
- supersede;
- expire;
- retract;
- dispute;
- accept;
- materialize.

These are knowledge-management operations.

## Relationship to existing systems

### TerminusDB

TerminusDB is the closest conceptual relative. It already has immutable commits, branches, structured diffs, triple-level merges, time travel, clone, push, and pull. Its documentation describes it as Git-like version control for facts rather than files. See [TerminusDB version control](https://terminusdb.org/docs/knowledge-graph-version-control/).

Before writing a new storage engine, it would be sensible to prototype an OKF repository backed by TerminusDB and learn which requirements it does not satisfy.

Potential mismatches:

- it is graph and database first;
- OKF is Markdown and document first;
- natural-language claims do not always map cleanly to triples;
- its history model is not automatically the same as business-valid time;
- it introduces a database service and schema model.

### Dolt

Dolt proves that branch, merge, diff, and time travel can be adapted to structured relational data. Its merges operate at cell level rather than text-line level. See the [Dolt repository](https://github.com/dolthub/dolt).

Potential mismatches:

- knowledge is not naturally tabular;
- prose and citations need auxiliary storage;
- semantic contradictions remain application-level concerns.

### Event sourcing

Event sourcing may be the best architectural foundation. An append-only event log records what happened, while materialized views provide current queryable state. See the [Event Sourcing pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/event-sourcing).

The mapping is almost exact:

```text
event store       → knowledge events and changesets
materialized view → current OKF bundle
alternate view    → historical or valid-at OKF bundle
read model        → lexical, graph, or vector runtime indexes
```

This avoids reinventing Git's object storage while giving the domain a natural temporal model.

## Should this be an extension of Git?

Git can be extended with:

- custom diff drivers;
- custom merge drivers;
- hooks;
- commit trailers;
- Git notes;
- clean and smudge filters;
- an external semantic index;
- a domain-specific CLI wrapping Git.

This could produce a useful prototype:

```text
okfv commit
```

might:

1. inspect the Markdown diff;
2. extract semantic changes;
3. validate evidence and time ranges;
4. write a structured changeset;
5. create the underlying Git commit;
6. update the temporal index.

That reuses Git for storage and transport while adding knowledge semantics.

Git's fundamental objects would still remain:

```text
blob → tree → commit
```

The system would reconstruct semantic changes after the fact. It would not natively store assertions, validity intervals, or evidence dependencies.

Therefore:

- extend Git if the main objective is a fast prototype and human collaboration;
- build a temporal ledger if semantic time and automated maintenance are central;
- build a new low-level VCS only if distribution, branching, and repository scale cannot be supported by existing storage.

The third condition is unlikely to be true initially.

## A pragmatic architecture

A first implementation could use SQLite rather than a new object database.

```text
repository/
├── bundle/                 # Materialized current OKF
├── sources/                # Or external content-addressed source store
└── .okfv/
    ├── ledger.sqlite       # Events, claims, validity, and changesets
    ├── objects/            # Large content-addressed blobs if needed
    ├── config.yaml
    └── indexes/            # Disposable search indexes
```

Possible tables:

```text
repository_revisions
changesets
events
source_revisions
concepts
concept_revisions
claims
claim_evidence
relationships
conflicts
materializations
```

The ledger can be append-only at the application level. SQLite transactions provide atomicity. Each accepted changeset generates a new repository revision and bundle fingerprint.

Git may remain an optional transport and collaboration backend:

```text
okfv export-git
okfv import-git
okfv publish
```

If the experiment succeeds, the storage layer can later become distributed or content-addressed without changing the temporal semantics.

## Risks of creating a new system

### Scope explosion

The project could easily grow into:

- a version-control system;
- a temporal database;
- a knowledge graph;
- a document editor;
- a workflow engine;
- a provenance system;
- a RAG platform.

That would be too much.

The first version should avoid:

- network synchronization;
- distributed merges;
- arbitrary ontology reasoning;
- collaborative real-time editing;
- binary storage;
- automatic contradiction resolution;
- custom query languages.

### False semantic precision

An LLM converting prose into claims can be wrong. A structured event is not necessarily true merely because it is structured.

Every derived claim must preserve:

- exact evidence;
- producer identity;
- extraction method;
- uncertainty;
- review state.

### Dual sources of truth

If users edit Markdown while the ledger changes independently, they will diverge.

Every mutation needs one controlled path:

```text
edit or ingest
    ↓
proposed semantic changeset
    ↓
accept
    ↓
ledger update
    ↓
OKF rematerialization
```

Direct edits can be supported, but only by importing them as proposals.

### Portability

Core OKF consumers see only the materialized snapshot, not the ledger history. This is acceptable if the snapshot remains fully conformant and the history format is documented as an optional extension.

### Deletion and privacy

An append-only ledger preserves removed knowledge. That conflicts with privacy deletion, secret removal, and some content licenses. The design needs redaction events and a destructive administrative purge process, even if ordinary history is immutable.

## A focused MVP

Do not begin with branches, remotes, or Git compatibility.

### Phase 1: temporal concept repository

Implement:

- stable concept identity;
- immutable source revisions;
- concept-level changesets;
- valid time and recorded time;
- supersede, expire, retract, and correct operations;
- current, valid-at, and known-at materialization;
- conformant OKF export;
- semantic history for one concept.

No LLM is required.

### Phase 2: claim and evidence layer

Add:

- stable claim IDs;
- exact evidence mappings;
- overlapping-validity conflict detection;
- semantic diff;
- temporal linting;
- source-change impact analysis.

### Phase 3: controlled LLM proposals

Let an LLM propose, but not directly apply:

- new claims;
- corrections;
- supersession;
- concept merges and splits;
- validity intervals.

Measure proposal accuracy and reviewer workload.

### Phase 4: branching and collaboration

Only after the semantic model is stable, add:

- proposal branches;
- three-way semantic merge;
- review workflow;
- optional Git import and export;
- signed acceptance records.

### Phase 5: distributed repository behavior

Clone, push, pull, content-addressed synchronization, and remote protocols should be considered only if real use cases require them.

## The decisive experiment

Before building substantial infrastructure, create a small evolving marketing corpus:

```text
T1: Offer A introduced.
T2: Offer A terms changed prospectively.
T3: A correction arrives late and applies retroactively.
T4: Offer A expires.
T5: Offer B replaces it for new customers.
T6: Offer A remains valid for existing contracts.
```

Require the prototype to answer:

- What is valid now?
- What was valid at T3?
- What did the KB believe at T3?
- When did the KB learn the correction?
- Why was a claim superseded?
- Which source supported each version?
- What changed semantically between two revisions?
- Can the bundle as known at T2 be reproduced exactly?
- Does a normal current-time query ever retrieve an expired offer?

If a Git wrapper cannot answer these cleanly but the ledger prototype can, the case for the new system is established.

## Honest recommendation

There is a real idea here, but it should be framed carefully.

Do not initially describe it as:

> A new version-control system replacing Git.

Describe it as:

> A temporal, evidence-aware knowledge repository that materializes OKF bundles and borrows the best ideas from version control.

That distinction keeps the work focused on the unsolved part:

- knowledge validity;
- provenance;
- supersession;
- semantic conflict;
- temporal retrieval;
- explainable evolution.

The storage engine, hashing algorithm, remote protocol, and branch mechanics are not the innovation. TerminusDB, Dolt, Git, SQLite, and event stores already provide strong building blocks.

Start with an event-sourced temporal ledger and a conformant OKF materialized view. Add Git interoperability before considering a custom Git-like engine.

If that prototype proves useful, it could become meaningfully original: not “Git for Markdown,” but a transparent temporal knowledge compiler.
