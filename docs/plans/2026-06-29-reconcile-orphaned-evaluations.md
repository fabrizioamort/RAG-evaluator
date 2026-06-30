# Reconcile Orphaned Evaluations on Startup

**Goal:** On backend startup, reconcile evaluations left stuck in a non-terminal
state by a crash or restart (`running`/`pending`) so they surface as recoverable
`failed` and can be resumed via the existing Retry button — with no manual DB
edit and no re-running of already-completed test cases.

**Architecture:** Mirror the existing `IndexBuildService.reconcile_interrupted_builds()`
pattern (called from the FastAPI lifespan startup, `main.py:71`). Add an
equivalent `reconcile_orphaned_evaluations()` to `JobCheckpointService` (which
already owns evaluation/job lifecycle transitions) and invoke it once at startup.

**Tech Stack:** FastAPI lifespan, SQLAlchemy async, SQLite/Postgres, pytest.

---

## Root Cause (proven)

- Evaluations run as **in-process asyncio background tasks** (`_run_evaluation_background`
  → `EvaluationRunner.run()`). A crash/restart kills the event loop, so every
  in-flight task vanishes, but the DB row keeps `status='running'` and
  `evaluation_jobs.state='running'`.
- The retry endpoint (`app/api/evaluations.py`) **rejects `running`** — it only
  accepts `failed`, `cancelled`, or an incomplete `completed`. So a crashed run
  is unrecoverable from the UI: it looks alive but nothing is executing.
- **Resume already works** once the status is recoverable: the runner skips test
  cases that already have a saved result
  (`_get_completed_test_case_ids`, `evaluation_runner.py:748`) and only processes
  `remaining_test_cases` (`evaluation_runner.py:345`). Retry therefore resumes
  from the next unfinished case with zero rework.
- Confirmed live example: evaluation `4d5e5ad0…` was stuck `running` with 49/100
  `evaluation_results` rows persisted and a frozen heartbeat.

## Design Decisions

1. **Unconditional reconcile (recommended).** The deployment is single-process
   (`dev_server.py`, single uvicorn worker). At startup there can be no live
   evaluation task, so **any** evaluation still in `running` or `pending` is, by
   definition, orphaned. Reconcile them unconditionally — no need to wait for a
   heartbeat-staleness window to elapse (which would leave an eval stuck if the
   process restarts inside the window). A heartbeat-guarded variant is documented
   below for a future multi-worker deployment.
2. **Target states:** `running` and `pending` → `failed` (recoverable). Including
   `pending` is safe because the server has not accepted any request yet at
   startup, so a `pending` row can only be from a previous process.
3. **Leave `paused` alone.** It is a deliberate user state; the existing resume
   endpoint already re-runs paused evaluations after a restart.
4. **Leave terminal states alone** (`completed`, `failed`, `cancelled`).
5. **Recoverable error message,** mirroring the index reconciler's wording so the
   UI clearly signals "retry to resume".
6. **Keep `evaluation_jobs.state` consistent** — set it to `failed` too, matching
   `JobCheckpointService.fail_job()`.

**No DB migration** (writes only existing columns: `evaluations.status`,
`evaluations.error_message`, `evaluations.completed_at`, `evaluation_jobs.state`,
`evaluation_jobs.error_message`, `evaluation_jobs.last_heartbeat`).
**No frontend change** — the Retry UI already shipped (commit `854c415`); it
renders for `failed` evaluations.

---

### Task 1: Add `reconcile_orphaned_evaluations()` to JobCheckpointService

**Files:**
- Modify: `platform/backend/app/services/job_checkpoint_service.py`

**Step 1: Add the method** (after `fail_job`):

```python
async def reconcile_orphaned_evaluations(self) -> int:
    """Mark evaluations interrupted by a restart as failed-but-recoverable.

    At startup no evaluation task can be live, so any evaluation still in an
    active, non-terminal state was orphaned by the previous process. Retrying
    resumes them, skipping test cases that already have a saved result.

    Returns:
        Number of evaluations reconciled.
    """
    now = datetime.now(timezone.utc)
    message = (
        "Evaluation was interrupted by a backend restart and can be "
        "resumed by retrying."
    )
    result = await self.db.execute(
        select(Evaluation).where(Evaluation.status.in_(["running", "pending"]))
    )
    evaluations = result.scalars().all()
    for evaluation in evaluations:
        evaluation.status = "failed"
        evaluation.error_message = message
        evaluation.completed_at = now
    if evaluations:
        eval_ids = [e.id for e in evaluations]
        await self.db.execute(
            update(EvaluationJob)
            .where(EvaluationJob.evaluation_id.in_(eval_ids))
            .values(state="failed", error_message=message, last_heartbeat=now)
        )
        await self.db.commit()
    return len(evaluations)
```

`select`, `update`, `Evaluation`, `EvaluationJob`, `datetime`, `timezone` are
already imported in this module — no new imports needed.

**Step 2 (optional, multi-worker future):** to make it heartbeat-safe for >1
worker, add a `stale_after: timedelta | None = None` parameter and, when set,
filter to evaluations whose `evaluation_jobs.last_heartbeat` is older than
`now - stale_after` (join `EvaluationJob`), exactly like
`reconcile_interrupted_builds(stale_after=...)`. Default `None` keeps the
single-process unconditional behavior. Do **not** add this now unless multi-worker
is on the roadmap (keep it simple).

### Task 2: Invoke it from the lifespan startup

**Files:**
- Modify: `platform/backend/app/main.py`

**Step 1:** Inside the existing `async with get_db_context() as db:` block in
`lifespan()` (right after the `reconcile_interrupted_builds()` call, ~`main.py:73`):

```python
from app.services.job_checkpoint_service import get_checkpoint_service

orphaned = await get_checkpoint_service(db).reconcile_orphaned_evaluations()
if orphaned:
    logger.warning("Marked orphaned evaluations as failed", count=orphaned)
```

Reuse the same `db` session already opened for the template/index reconcile.
Startup runs before the server accepts traffic, so there is no race with new
evaluation creation.

### Task 3: Tests

**Files:**
- Modify: `platform/backend/tests/test_services/test_evaluation_lifecycle.py`
  (or add `test_reconcile_orphaned_evaluations.py` alongside it)

**Cases to cover** (in-memory SQLite session, seed rows directly — do **not**
import `EvaluationRunner`/`deepeval`, see verification note):

1. `running` eval with 2 of 3 `evaluation_results` rows → after reconcile:
   `status == "failed"`, `error_message` set, and the 2 result rows still exist
   (asserts **no data loss** — the core promise of resume).
2. `pending` eval → reconciled to `failed`.
3. `paused` eval → untouched.
4. `completed` and `failed` evals → untouched (status unchanged).
5. Linked `evaluation_jobs.state` flips to `failed` and `last_heartbeat` updates.
6. Return value equals the number of reconciled evaluations.

### Task 4: Verify end state

**Step 1 — focused local verification (avoids the known pytest hang).**
`uv run pytest` hangs on this machine because importing the API/runner pulls
`deepeval`/`chromadb` at collection time. Verify with a standalone in-memory
SQLite harness that imports only the models + `JobCheckpointService`:

```powershell
cd platform/backend
uv run python -c "import asyncio; from tests.helpers.reconcile_check import main; asyncio.run(main())"
```

(or an inline script in the scratchpad that: creates an in-memory engine, seeds a
`running` eval + partial results + job, calls `reconcile_orphaned_evaluations()`,
and asserts status/results). This path never imports `deepeval`.

Expected: reconciled count `1`, status `failed`, partial results preserved.

**Step 2 — syntax check:**

```powershell
cd platform/backend
uv run python -m py_compile app/services/job_checkpoint_service.py app/main.py
```

Expected: no output, exit code `0`.

**Step 3 — manual end-to-end (optional):** start the backend with an evaluation
row forced to `running`; confirm the startup log emits
`Marked orphaned evaluations as failed count=1`, the eval shows `failed` in the
UI, and clicking **Retry** resumes from the first unfinished test case (completed
results are not recomputed).

---

## Out of Scope / Notes

- **Auto-resume on startup** (re-queuing the background task automatically) is
  intentionally excluded — it risks a thundering herd and surprising token spend.
  Marking `failed`-recoverable + manual Retry mirrors the index-build behavior.
- The three older pre-existing stuck `running` evaluations will be cleaned up the
  first time this ships (they meet the reconcile criteria).
