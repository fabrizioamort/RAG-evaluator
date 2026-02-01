"""SQLAlchemy models for the RAG Evaluation Platform."""

from app.database import Base
from app.models.artifact import Artifact
from app.models.base import BaseModel, BaseModelNoUpdate
from app.models.comparison import Comparison
from app.models.document import Document
from app.models.evaluation import Evaluation
from app.models.evaluation_job import EvaluationJob
from app.models.evaluation_result import EvaluationResult
from app.models.knowledge_base import KnowledgeBase
from app.models.knowledge_base_index import KnowledgeBaseIndex
from app.models.knowledge_base_version import KnowledgeBaseVersion
from app.models.project import Project
from app.models.rag_config import RAGConfig
from app.models.run_manifest import RunManifest
from app.models.test_case import TestCase
from app.models.test_generation_job import TestGenerationJob
from app.models.test_set import TestSet
from app.models.test_template import TestTemplate
from app.models.playground_query import PlaygroundQuery
from app.models.webhook import Webhook

__all__ = [
    "Base",
    "BaseModel",
    "BaseModelNoUpdate",
    "Project",
    "KnowledgeBase",
    "KnowledgeBaseIndex",
    "KnowledgeBaseVersion",
    "Document",
    "TestTemplate",
    "TestSet",
    "TestCase",
    "TestGenerationJob",
    "RAGConfig",
    "Artifact",
    "RunManifest",
    "Evaluation",
    "EvaluationJob",
    "EvaluationResult",
    "Comparison",
    "Webhook",
    "PlaygroundQuery",
]
