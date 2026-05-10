# Kill Backend Target Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `make kill-backend` target that stops only processes listening on port `8000`, working on both Windows and Linux.

**Architecture:** Reuse the tracked backend launcher script as the single place for backend process management. Extend it with cross-platform listener discovery and a kill-only CLI mode, then route the new `Makefile` target through that script to avoid OS-specific shell recipes in the `Makefile`.

**Tech Stack:** GNU Make, Python stdlib (`subprocess`, `os`, `signal`, `time`), `unittest`

---

### Task 1: Lock in the new target with tests

**Files:**
- Modify: `C:\Users\fabri\projects\RAG-evaluator\tests\unit\test_makefile_dev_backend.py`
- Modify: `C:\Users\fabri\projects\RAG-evaluator\tests\unit\test_backend_dev_server.py`

**Step 1: Write the failing test**

Add a `Makefile` assertion for `kill-backend` and unit coverage for the backend launcher’s cross-platform listener lookup and kill-only flow.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.unit.test_makefile_dev_backend tests.unit.test_backend_dev_server -v`
Expected: failure because `kill-backend` does not exist and the launcher has no kill-only mode yet.

**Step 3: Write minimal implementation**

Add the new target and only the launcher functions needed to make the tests pass.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.unit.test_makefile_dev_backend tests.unit.test_backend_dev_server -v`
Expected: PASS

### Task 2: Wire the new target into developer help

**Files:**
- Modify: `C:\Users\fabri\projects\RAG-evaluator\Makefile`

**Step 1: Update help and phony targets**

Add `kill-backend` to `.PHONY` and the help output.

**Step 2: Keep the target narrow**

Ensure the target delegates to the backend launcher with a kill-only mode for port `8000`, with no wildcard Python process matching.

**Step 3: Verify manually**

Run: `python -m py_compile platform/backend/dev_server.py tests/unit/test_makefile_dev_backend.py tests/unit/test_backend_dev_server.py`
Expected: no output, exit code `0`

### Task 3: Verify end state

**Files:**
- Modify: `C:\Users\fabri\projects\RAG-evaluator\platform\backend\dev_server.py`
- Modify: `C:\Users\fabri\projects\RAG-evaluator\Makefile`
- Test: `C:\Users\fabri\projects\RAG-evaluator\tests\unit\test_makefile_dev_backend.py`
- Test: `C:\Users\fabri\projects\RAG-evaluator\tests\unit\test_backend_dev_server.py`

**Step 1: Run focused verification**

Run: `python -m unittest tests.unit.test_makefile_dev_backend tests.unit.test_backend_dev_server -v`
Expected: PASS

**Step 2: Run syntax verification**

Run: `python -m py_compile platform/backend/dev_server.py tests/unit/test_makefile_dev_backend.py tests/unit/test_backend_dev_server.py`
Expected: no output, exit code `0`
