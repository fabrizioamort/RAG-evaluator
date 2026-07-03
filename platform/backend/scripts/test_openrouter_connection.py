"""Minimal OpenRouter connectivity test.

Sends "hi" to deepseek/deepseek-v4-flash and prints either the answer or the
HTTP/client error. Reads OPENROUTER_API_KEY first, then OPENAI_API_KEY.

Usage:
    uv run python scripts/test_openrouter_connection.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v4-flash"


def load_dotenv() -> None:
    """Load simple KEY=VALUE entries from platform/backend/.env if present."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY or OPENAI_API_KEY.", file=sys.stderr)
        return 2

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0,
        "max_tokens": 64,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "RAG Evaluator OpenRouter Test",
        },
    )

    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            elapsed = time.time() - started
            raw = response.read().decode("utf-8")
            data = json.loads(raw)
            print(f"HTTP {response.status} in {elapsed:.2f}s")
            print(f"model: {data.get('model')}")
            print(f"id: {data.get('id')}")
            print("answer:")
            print(data["choices"][0]["message"].get("content", ""))
            return 0
    except urllib.error.HTTPError as exc:
        elapsed = time.time() - started
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP ERROR {exc.code} in {elapsed:.2f}s", file=sys.stderr)
        print(error_body, file=sys.stderr)
        return 1
    except Exception as exc:
        elapsed = time.time() - started
        print(f"CLIENT ERROR after {elapsed:.2f}s: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
