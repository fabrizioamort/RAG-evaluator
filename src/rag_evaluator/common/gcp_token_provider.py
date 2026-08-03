"""GCP access token provider for Vertex AI OpenAI-compatible endpoint.

Vertex AI exposes an OpenAI-compatible chat completions / embeddings endpoint at
``https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/endpoints/openapi``.
Authentication uses a Google Cloud OAuth2 access token in the ``Authorization: Bearer``
header. Tokens are obtained from Application Default Credentials (ADC) and expire
after ~1 hour, so this module exposes a thread-safe provider with proactive refresh
plus an ``httpx.Auth`` hook that injects a fresh token on every request.

Optional dependency: ``google-auth`` (installed with the ``google-vertex`` extra).
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, cast

import httpx

try:
    from google.auth import default as _google_auth_default
    from google.auth.transport.requests import Request as _GoogleAuthRequest

    GOOGLE_AUTH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when optional extra is missing
    _google_auth_default = None  # type: ignore[assignment]
    _GoogleAuthRequest = None  # type: ignore[assignment,misc]
    GOOGLE_AUTH_AVAILABLE = False

if TYPE_CHECKING:
    from collections.abc import Generator


GOOGLE_AUTH_INSTALL_HINT = (
    "Google Vertex AI Gemini support requires 'google-auth'. "
    "Install it with: uv sync --extra google-vertex"
)

# GCP scope required for Vertex AI generative endpoints
_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

# Renew the token proactively when fewer than this many seconds remain
_REFRESH_THRESHOLD_SECONDS = 300  # 5 minutes


class _Credentials(Protocol):
    """Subset of google-auth credentials used by :class:`GCPTokenProvider`."""

    token: str | None
    expiry: datetime | None

    def refresh(self, request: object) -> None: ...


def require_google_auth() -> None:
    """Raise ImportError with an actionable message if google-auth is missing."""
    if not GOOGLE_AUTH_AVAILABLE:
        raise ImportError(GOOGLE_AUTH_INSTALL_HINT)


class GCPTokenProvider:
    """Thread-safe access token provider with proactive refresh.

    Prefer :meth:`instance` over direct construction — the ADC lookup and
    credentials refresh have non-trivial cost and should be shared across the
    process.
    """

    _instance: GCPTokenProvider | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        require_google_auth()
        assert _google_auth_default is not None  # narrowing for mypy
        self._creds, self._detected_project = cast(
            tuple[_Credentials, str | None],
            _google_auth_default(scopes=[_SCOPE]),
        )
        self._lock = threading.Lock()

    @classmethod
    def instance(cls) -> GCPTokenProvider:
        """Return the process-wide provider (lazy init, thread-safe)."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear the cached instance (test-only)."""
        with cls._instance_lock:
            cls._instance = None

    @property
    def detected_project(self) -> str | None:
        """The GCP project auto-detected by ADC (may be ``None``)."""
        return self._detected_project

    def _needs_refresh(self) -> bool:
        if not self._creds.token:
            return True
        if self._creds.expiry is None:
            return True
        # google-auth stores expiry as a naive UTC datetime
        expiry_aware = self._creds.expiry.replace(tzinfo=UTC)
        remaining = (expiry_aware - datetime.now(UTC)).total_seconds()
        return remaining < _REFRESH_THRESHOLD_SECONDS

    def get_token(self) -> str:
        """Return a valid access token, refreshing if needed."""
        with self._lock:
            if self._needs_refresh():
                assert _GoogleAuthRequest is not None  # narrowing for mypy
                self._creds.refresh(_GoogleAuthRequest())
        token = self._creds.token
        if not token:
            raise RuntimeError(
                "GCP access token is empty after refresh — check ADC configuration."
            )
        return str(token)


class GCPBearerAuth(httpx.Auth):
    """httpx Auth flow that injects a fresh GCP bearer token on every request."""

    requires_response_body = False

    def __init__(self, provider: GCPTokenProvider | None = None) -> None:
        self._provider = provider or GCPTokenProvider.instance()

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["Authorization"] = f"Bearer {self._provider.get_token()}"
        yield request


def build_vertex_openai_base_url(project: str, location: str) -> str:
    """Return the Vertex AI OpenAI-compatible endpoint base URL."""
    if not project:
        raise ValueError(
            "GOOGLE_CLOUD_PROJECT is not set — required for Vertex AI Gemini."
        )
    if not location:
        raise ValueError("GOOGLE_CLOUD_LOCATION is not set — required for Vertex AI Gemini.")
    return (
        f"https://{location}-aiplatform.googleapis.com/v1beta1/"
        f"projects/{project}/locations/{location}/endpoints/openapi"
    )


def prepend_google_prefix(model: str) -> str:
    """Prepend ``google/`` to a model name if not already namespaced.

    The Vertex OpenAI-compat endpoint requires ``google/gemini-2.5-flash``,
    not ``gemini-2.5-flash``. Passes through names that already contain a slash
    (e.g. ``google/gemini-2.5-pro``, ``publishers/meta/...``).
    """
    if not model:
        return model
    if "/" in model:
        return model
    return f"google/{model}"
