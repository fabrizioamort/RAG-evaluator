"""Auth resolution and client factory for Google Vertex AI Search (Discovery Engine).

The `google-cloud-discoveryengine` / `google-auth` / `google-cloud-storage` packages
are an optional extra (`uv sync --extra google-vertex`). Imports are guarded so the
registry stays importable even when the extra is not installed; constructing a
``GoogleVertexSearchRAG`` instance raises a friendly error in that case.
"""

from __future__ import annotations

from typing import Any

try:
    from google.api_core.exceptions import NotFound
    from google.cloud import discoveryengine_v1 as discoveryengine
    from google.oauth2 import service_account

    GOOGLE_VERTEX_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when optional extra is missing
    discoveryengine = None  # type: ignore[assignment]
    service_account = None  # type: ignore[assignment]
    NotFound = Exception  # type: ignore[assignment, misc]
    GOOGLE_VERTEX_AVAILABLE = False

from rag_evaluator.config import settings

GOOGLE_VERTEX_INSTALL_HINT = (
    "Google Vertex AI Search support requires the optional 'google-vertex' extra. "
    "Install it with: uv sync --extra google-vertex"
)

DEFAULT_COLLECTION = "default_collection"
DEFAULT_BRANCH = "default_branch"
DEFAULT_SERVING_CONFIG = "default_search"


def require_google_vertex() -> None:
    """Raise ImportError with an actionable message if the GCP libs are missing."""
    if not GOOGLE_VERTEX_AVAILABLE:
        raise ImportError(GOOGLE_VERTEX_INSTALL_HINT)


def resolve_credentials(sa_key_path: str | None = None) -> Any | None:
    """Resolve credentials: explicit service-account JSON, else ADC (None)."""
    require_google_vertex()
    resolved = sa_key_path or settings.google_vertex_sa_key_path
    if resolved:
        return service_account.Credentials.from_service_account_file(resolved)
    return None


def _client_kwargs(location: str, sa_key_path: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    credentials = resolve_credentials(sa_key_path)
    if credentials is not None:
        kwargs["credentials"] = credentials
    if location != "global":
        kwargs["client_options"] = {"api_endpoint": f"{location}-discoveryengine.googleapis.com"}
    return kwargs


def get_data_store_service_client(
    location: str, sa_key_path: str | None = None
) -> discoveryengine.DataStoreServiceClient:
    """Build a DataStoreServiceClient for the given region."""
    require_google_vertex()
    return discoveryengine.DataStoreServiceClient(**_client_kwargs(location, sa_key_path))


def get_document_service_client(
    location: str, sa_key_path: str | None = None
) -> discoveryengine.DocumentServiceClient:
    """Build a DocumentServiceClient for the given region."""
    require_google_vertex()
    return discoveryengine.DocumentServiceClient(**_client_kwargs(location, sa_key_path))


def get_search_service_client(
    location: str, sa_key_path: str | None = None
) -> discoveryengine.SearchServiceClient:
    """Build a SearchServiceClient for the given region."""
    require_google_vertex()
    return discoveryengine.SearchServiceClient(**_client_kwargs(location, sa_key_path))


def get_conversational_search_service_client(
    location: str, sa_key_path: str | None = None
) -> discoveryengine.ConversationalSearchServiceClient:
    """Build a ConversationalSearchServiceClient (for grounded answers)."""
    require_google_vertex()
    return discoveryengine.ConversationalSearchServiceClient(**_client_kwargs(location, sa_key_path))


def collection_path(project_id: str, location: str) -> str:
    """Return the resource path of the default collection."""
    return f"projects/{project_id}/locations/{location}/collections/{DEFAULT_COLLECTION}"


def data_store_path(project_id: str, location: str, data_store_id: str) -> str:
    """Return the resource path of a data store."""
    return f"{collection_path(project_id, location)}/dataStores/{data_store_id}"


def branch_path(project_id: str, location: str, data_store_id: str) -> str:
    """Return the resource path of the default branch of a data store."""
    return f"{data_store_path(project_id, location, data_store_id)}/branches/{DEFAULT_BRANCH}"


def serving_config_path(project_id: str, location: str, data_store_id: str) -> str:
    """Return the resource path of the default search serving config."""
    return (
        f"{data_store_path(project_id, location, data_store_id)}"
        f"/servingConfigs/{DEFAULT_SERVING_CONFIG}"
    )


def validate_project_config(project_id: str, location: str) -> None:
    """Raise ValueError with a clear message if required GCP config is missing."""
    if not project_id:
        raise ValueError(
            "GOOGLE_VERTEX_PROJECT_ID is not set. Configure it in .env or pass it via "
            "the RAG config parameters."
        )
    if not location:
        raise ValueError("GOOGLE_VERTEX_LOCATION is not set (expected 'global', 'us', or 'eu').")
