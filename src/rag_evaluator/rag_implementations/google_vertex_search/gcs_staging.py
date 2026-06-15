"""Stage local documents into Google Cloud Storage for Vertex AI Search import.

Vertex AI Search imports unstructured documents from GCS (or BigQuery / inline),
not from local disk. This module uploads files discovered by
:func:`rag_evaluator.common.indexing.discover_source_documents` into a staging
bucket so they can be referenced by ``gcs_source`` in ``import_documents``.
"""

from __future__ import annotations

try:
    from google.cloud import storage

    GOOGLE_STORAGE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when optional extra is missing
    storage = None  # type: ignore[assignment]
    GOOGLE_STORAGE_AVAILABLE = False

from rag_evaluator.config import settings
from rag_evaluator.rag_implementations.google_vertex_search.client import resolve_credentials

GOOGLE_STORAGE_INSTALL_HINT = (
    "Google Cloud Storage support requires the optional 'google-vertex' extra. "
    "Install it with: uv sync --extra google-vertex"
)


def require_google_storage() -> None:
    """Raise ImportError with an actionable message if google-cloud-storage is missing."""
    if not GOOGLE_STORAGE_AVAILABLE:
        raise ImportError(GOOGLE_STORAGE_INSTALL_HINT)


def get_storage_client(sa_key_path: str | None = None) -> storage.Client:
    """Build a GCS client using the resolved credentials (SA key or ADC)."""
    require_google_storage()
    credentials = resolve_credentials(sa_key_path)
    if credentials is not None:
        return storage.Client(project=settings.google_vertex_project_id, credentials=credentials)
    return storage.Client(project=settings.google_vertex_project_id)


def gcs_blob_name(prefix: str, relative_path: str) -> str:
    """Return the blob name for a document under the given staging prefix."""
    return f"{prefix}/{relative_path}"


def gcs_uri(bucket_name: str, prefix: str, relative_path: str) -> str:
    """Return the ``gs://`` URI for a document under the given staging prefix."""
    return f"gs://{bucket_name}/{gcs_blob_name(prefix, relative_path)}"


def upload_file(
    client: storage.Client,
    bucket_name: str,
    source_path: str,
    blob_name: str,
) -> str:
    """Upload a local file to GCS, returning its ``gs://`` URI."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(source_path)
    return f"gs://{bucket_name}/{blob_name}"


def blob_exists(client: storage.Client, bucket_name: str, blob_name: str) -> bool:
    """Return True if the given blob already exists in the bucket."""
    bucket = client.bucket(bucket_name)
    return bool(bucket.blob(blob_name).exists())
