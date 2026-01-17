"""Services for the RAG Evaluation Platform."""

from app.services.artifact_store import ArtifactStore, get_artifact_store
from app.services.evaluation_runner import EvaluationRunner, get_evaluation_runner
from app.services.index_build_service import IndexBuildService, get_index_build_service
from app.services.job_checkpoint_service import JobCheckpointService, get_checkpoint_service
from app.services.job_event_log import JobEventLog, get_job_event_log
from app.services.rag_adapter import RAGAdapterService, get_rag_adapter_service
from app.services.storage_service import StorageService, get_storage_service
from app.services.test_generator_service import (
    GeneratedTestCase,
    GenerationConfig,
    GenerationProgress,
    TestGeneratorService,
    get_test_generator_service,
)
from app.services.test_quality_gate import (
    QualityConfig,
    QualityResult,
    TestQualityGateService,
    get_quality_gate_service,
)

__all__ = [
    "ArtifactStore",
    "get_artifact_store",
    "StorageService",
    "get_storage_service",
    "JobEventLog",
    "get_job_event_log",
    "JobCheckpointService",
    "get_checkpoint_service",
    "EvaluationRunner",
    "get_evaluation_runner",
    # Index building
    "IndexBuildService",
    "get_index_build_service",
    # RAG adapter
    "RAGAdapterService",
    "get_rag_adapter_service",
    # Test generation
    "TestGeneratorService",
    "get_test_generator_service",
    "GenerationConfig",
    "GenerationProgress",
    "GeneratedTestCase",
    # Quality gate
    "TestQualityGateService",
    "get_quality_gate_service",
    "QualityConfig",
    "QualityResult",
]
