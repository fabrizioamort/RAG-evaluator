"""Pytest bootstrap defaults for stable local and CI runs."""

import os
from pathlib import Path

# Prevent DeepEval from spawning telemetry/background behavior during tests.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_ASYNC_MODE", "False")

# Keep temporary files inside the workspace when system temp is restricted.
_LOCAL_TMP = Path(".pytest_tmp").resolve()
_LOCAL_TMP.mkdir(parents=True, exist_ok=True)
for _tmp_var in ("TMP", "TEMP", "TMPDIR"):
    os.environ.setdefault(_tmp_var, str(_LOCAL_TMP))
