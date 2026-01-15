"""Knowledge Bases API endpoints."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.api.deps import DbSession, Pagination
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.knowledge_base_version import KnowledgeBaseVersion
from app.models.project import Project
from app.models.rag_config import RAGConfig
from app.schemas.knowledge_base import (
    DocumentResponse,
    DocumentUploadResponse,
    KnowledgeBaseCreate,
    KnowledgeBaseIndexRequest,
    KnowledgeBaseList,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
    KnowledgeBaseVersionResponse,
    KnowledgeBaseWithDocuments,
)
from app.services.rag_adapter import RAGAdapterService, get_rag_adapter_service
from app.services.storage_service import StorageService, get_storage_service
from app.utils.logging_config import get_logger

router = APIRouter(tags=["Knowledge Bases"])
logger = get_logger(__name__)

# Type alias for storage service dependency
StorageDep = StorageService


def _kb_to_response(kb: KnowledgeBase) -> KnowledgeBaseResponse:
    """Convert KnowledgeBase model to KnowledgeBaseResponse schema."""
    return KnowledgeBaseResponse(
        id=kb.id,
        project_id=kb.project_id,
        name=kb.name,
        description=kb.description,
        metadata=kb.metadata_ if isinstance(kb.metadata_, dict) else {},
        status=kb.status,
        current_version=kb.current_version,
        storage_path=kb.storage_path,
        index_path=kb.index_path,
        document_count=len(kb.documents) if kb.documents else 0,
        created_at=kb.created_at,
    )


def _kb_to_response_with_documents(kb: KnowledgeBase) -> KnowledgeBaseWithDocuments:
    """Convert KnowledgeBase model to KnowledgeBaseWithDocuments schema."""
    return KnowledgeBaseWithDocuments(
        id=kb.id,
        project_id=kb.project_id,
        name=kb.name,
        description=kb.description,
        metadata=kb.metadata_ if isinstance(kb.metadata_, dict) else {},
        status=kb.status,
        current_version=kb.current_version,
        storage_path=kb.storage_path,
        index_path=kb.index_path,
        document_count=len(kb.documents) if kb.documents else 0,
        created_at=kb.created_at,
        documents=[
            DocumentResponse(
                id=doc.id,
                knowledge_base_id=doc.knowledge_base_id,
                filename=doc.filename,
                file_path=doc.file_path,
                content_type=doc.content_type,
                size_bytes=doc.size_bytes,
                checksum=doc.checksum,
                status=doc.status,
                created_at=doc.created_at,
            )
            for doc in kb.documents
        ]
        if kb.documents
        else [],
    )


def _document_to_response(doc: Document) -> DocumentResponse:
    """Convert Document model to DocumentResponse schema."""
    return DocumentResponse(
        id=doc.id,
        knowledge_base_id=doc.knowledge_base_id,
        filename=doc.filename,
        file_path=doc.file_path,
        content_type=doc.content_type,
        size_bytes=doc.size_bytes,
        checksum=doc.checksum,
        status=doc.status,
        created_at=doc.created_at,
    )


async def _get_project_or_404(db: DbSession, project_id: UUID) -> Project:
    """Get project by ID or raise 404."""
    query = select(Project).where(Project.id == project_id)
    result = await db.execute(query)
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id {project_id} not found",
        )
    return project


async def _get_kb_or_404(db: DbSession, kb_id: UUID) -> KnowledgeBase:
    """Get knowledge base by ID or raise 404."""
    query = (
        select(KnowledgeBase)
        .where(KnowledgeBase.id == kb_id)
        .options(selectinload(KnowledgeBase.documents))
    )
    result = await db.execute(query)
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Knowledge base with id {kb_id} not found",
        )
    return kb


async def _create_version(
    db: DbSession,
    kb: KnowledgeBase,
    change_type: str,
    change_description: str | None = None,
    documents: list[Document] | None = None,
) -> KnowledgeBaseVersion:
    """Create a new KB version snapshot.

    Args:
        db: Database session.
        kb: Knowledge base to version.
        change_type: Type of change (initial, documents_added, documents_removed, reindexed).
        change_description: Optional description of the change.
        documents: Optional list of documents to snapshot. If None, assumes empty list
                   (avoids lazy loading issues in async context).

    Returns:
        The created KnowledgeBaseVersion.
    """
    # Increment version number
    new_version_number = kb.current_version + 1

    # Create document snapshot from provided documents (avoids lazy loading)
    document_snapshot: list[dict[str, Any]] = []
    docs_to_snapshot = documents if documents is not None else []
    for doc in docs_to_snapshot:
        document_snapshot.append(
            {
                "id": str(doc.id),
                "filename": doc.filename,
                "checksum": doc.checksum,
                "size_bytes": doc.size_bytes,
            }
        )

    # Create version record
    version = KnowledgeBaseVersion(
        knowledge_base_id=kb.id,
        version_number=new_version_number,
        change_type=change_type,
        document_snapshot=document_snapshot,
        change_description=change_description,
    )
    db.add(version)

    # Update KB current version
    kb.current_version = new_version_number

    await db.flush()
    return version


# =============================================================================
# Knowledge Base CRUD Endpoints
# =============================================================================


@router.get(
    "/projects/{project_id}/knowledge-bases",
    response_model=KnowledgeBaseList,
    summary="List knowledge bases in a project",
    description="Retrieve a paginated list of knowledge bases for a specific project.",
)
async def list_knowledge_bases(
    db: DbSession,
    project_id: UUID,
    pagination: Pagination,
) -> KnowledgeBaseList:
    """List all knowledge bases in a project."""
    # Verify project exists
    await _get_project_or_404(db, project_id)

    # Build query
    query = (
        select(KnowledgeBase)
        .where(KnowledgeBase.project_id == project_id)
        .options(selectinload(KnowledgeBase.documents))
    )

    # Get total count
    count_query = (
        select(func.count())
        .select_from(KnowledgeBase)
        .where(KnowledgeBase.project_id == project_id)
    )
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Apply pagination and ordering
    query = query.order_by(KnowledgeBase.created_at.desc())
    query = query.offset(pagination.offset).limit(pagination.limit)

    # Execute query
    result = await db.execute(query)
    knowledge_bases = result.scalars().all()

    logger.info(
        "Listed knowledge bases",
        project_id=str(project_id),
        count=len(knowledge_bases),
        total=total,
    )

    return KnowledgeBaseList(
        items=[_kb_to_response(kb) for kb in knowledge_bases],
        offset=pagination.offset,
        limit=pagination.limit,
        total=total,
    )


@router.post(
    "/projects/{project_id}/knowledge-bases",
    response_model=KnowledgeBaseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a knowledge base",
    description="Create a new knowledge base within a project.",
)
async def create_knowledge_base(
    db: DbSession,
    project_id: UUID,
    kb_data: KnowledgeBaseCreate,
    storage: StorageDep = Depends(get_storage_service),
) -> KnowledgeBaseResponse:
    """Create a new knowledge base."""
    # Verify project exists
    await _get_project_or_404(db, project_id)

    # Create knowledge base
    kb = KnowledgeBase(
        project_id=project_id,
        name=kb_data.name,
        description=kb_data.description,
        metadata_=kb_data.metadata_,
        status="pending",
        current_version=0,
    )
    db.add(kb)
    await db.flush()  # Get the ID

    # Set storage paths
    kb.storage_path = str(storage.get_documents_path(kb.id))
    kb.index_path = str(storage.get_index_path(kb.id))

    # Create initial version (no documents yet)
    await _create_version(db, kb, "initial", "Knowledge base created", documents=[])

    await db.commit()
    await db.refresh(kb)

    logger.info(
        "Created knowledge base",
        kb_id=str(kb.id),
        project_id=str(project_id),
        name=kb.name,
    )

    return KnowledgeBaseResponse(
        id=kb.id,
        project_id=kb.project_id,
        name=kb.name,
        description=kb.description,
        metadata=kb.metadata_ if isinstance(kb.metadata_, dict) else {},
        status=kb.status,
        current_version=kb.current_version,
        storage_path=kb.storage_path,
        index_path=kb.index_path,
        document_count=0,
        created_at=kb.created_at,
    )


@router.get(
    "/knowledge-bases/{kb_id}",
    response_model=KnowledgeBaseWithDocuments,
    summary="Get knowledge base details",
    description="Retrieve details of a knowledge base including its documents.",
    responses={404: {"description": "Knowledge base not found"}},
)
async def get_knowledge_base(
    db: DbSession,
    kb_id: UUID,
) -> KnowledgeBaseWithDocuments:
    """Get a knowledge base by ID with documents."""
    kb = await _get_kb_or_404(db, kb_id)
    return _kb_to_response_with_documents(kb)


@router.put(
    "/knowledge-bases/{kb_id}",
    response_model=KnowledgeBaseResponse,
    summary="Update a knowledge base",
    description="Update knowledge base metadata (name, description).",
    responses={404: {"description": "Knowledge base not found"}},
)
async def update_knowledge_base(
    db: DbSession,
    kb_id: UUID,
    kb_data: KnowledgeBaseUpdate,
) -> KnowledgeBaseResponse:
    """Update a knowledge base."""
    kb = await _get_kb_or_404(db, kb_id)

    # Update only provided fields
    update_data = kb_data.model_dump(exclude_unset=True, by_alias=False)
    for field, value in update_data.items():
        setattr(kb, field, value)

    await db.commit()

    # Re-fetch with relationships
    kb = await _get_kb_or_404(db, kb_id)

    logger.info(
        "Updated knowledge base",
        kb_id=str(kb_id),
        updated_fields=list(update_data.keys()),
    )

    return _kb_to_response(kb)


@router.delete(
    "/knowledge-bases/{kb_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a knowledge base",
    description="Delete a knowledge base and all its documents.",
    responses={404: {"description": "Knowledge base not found"}},
)
async def delete_knowledge_base(
    db: DbSession,
    kb_id: UUID,
    storage: StorageDep = Depends(get_storage_service),
) -> None:
    """Delete a knowledge base and all its data."""
    kb = await _get_kb_or_404(db, kb_id)

    # Delete storage files
    await storage.delete_kb_storage(kb_id)

    # Delete from database (cascade deletes documents and versions)
    await db.delete(kb)
    await db.commit()

    logger.info("Deleted knowledge base", kb_id=str(kb_id))


# =============================================================================
# Document Management Endpoints
# =============================================================================


@router.post(
    "/knowledge-bases/{kb_id}/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload documents",
    description="Upload one or more documents to a knowledge base.",
    responses={404: {"description": "Knowledge base not found"}},
)
async def upload_documents(
    db: DbSession,
    kb_id: UUID,
    files: list[UploadFile] = File(..., description="Documents to upload"),
    storage: StorageDep = Depends(get_storage_service),
) -> DocumentUploadResponse:
    """Upload documents to a knowledge base."""
    kb = await _get_kb_or_404(db, kb_id)

    uploaded: list[DocumentResponse] = []
    failed: list[dict[str, str]] = []
    total_size = 0

    for file in files:
        try:
            # Read file content
            content = await file.read()
            if not content:
                failed.append({"filename": file.filename or "unknown", "error": "Empty file"})
                continue

            # Save file to storage
            file_path, checksum, size_bytes = await storage.save_file(
                kb_id=kb_id,
                filename=file.filename or "unnamed",
                content=content,
            )

            # Create document record
            doc = Document(
                knowledge_base_id=kb_id,
                filename=file.filename or "unnamed",
                file_path=str(file_path),
                content_type=file.content_type,
                size_bytes=size_bytes,
                checksum=checksum,
                status="uploaded",
            )
            db.add(doc)
            await db.flush()
            await db.refresh(doc)

            uploaded.append(_document_to_response(doc))
            total_size += size_bytes

        except Exception as e:
            logger.exception("Failed to upload document", filename=file.filename)
            failed.append({"filename": file.filename or "unknown", "error": str(e)})

    # If any documents were uploaded, create a new version
    if uploaded:
        # Refresh KB to get updated documents list
        await db.refresh(kb, ["documents"])
        await _create_version(
            db,
            kb,
            "documents_added",
            f"Added {len(uploaded)} document(s)",
            documents=list(kb.documents) if kb.documents else [],
        )

        # Update KB status if it was pending and now has documents
        if kb.status == "pending":
            kb.status = "ready"

    await db.commit()

    logger.info(
        "Uploaded documents",
        kb_id=str(kb_id),
        uploaded_count=len(uploaded),
        failed_count=len(failed),
        total_size=total_size,
    )

    return DocumentUploadResponse(
        uploaded=uploaded,
        failed=failed,
        total_size_bytes=total_size,
    )


@router.delete(
    "/knowledge-bases/{kb_id}/documents/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
    description="Remove a document from a knowledge base.",
    responses={404: {"description": "Document or knowledge base not found"}},
)
async def delete_document(
    db: DbSession,
    kb_id: UUID,
    doc_id: UUID,
    storage: StorageDep = Depends(get_storage_service),
) -> None:
    """Delete a document from a knowledge base."""
    # Verify KB exists
    kb = await _get_kb_or_404(db, kb_id)

    # Find the document
    query = select(Document).where(Document.id == doc_id, Document.knowledge_base_id == kb_id)
    result = await db.execute(query)
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id {doc_id} not found in knowledge base {kb_id}",
        )

    # Delete file from storage
    await storage.delete_file(doc.file_path)

    # Delete from database
    await db.delete(doc)

    # Create version for document removal
    await db.refresh(kb, ["documents"])
    await _create_version(
        db,
        kb,
        "documents_removed",
        f"Removed document: {doc.filename}",
        documents=list(kb.documents) if kb.documents else [],
    )

    await db.commit()

    logger.info("Deleted document", kb_id=str(kb_id), doc_id=str(doc_id))


# =============================================================================
# Version Management Endpoints
# =============================================================================


@router.get(
    "/knowledge-bases/{kb_id}/versions",
    response_model=list[KnowledgeBaseVersionResponse],
    summary="List KB versions",
    description="Get version history for a knowledge base.",
    responses={404: {"description": "Knowledge base not found"}},
)
async def list_kb_versions(
    db: DbSession,
    kb_id: UUID,
) -> list[KnowledgeBaseVersionResponse]:
    """List all versions of a knowledge base."""
    # Verify KB exists
    await _get_kb_or_404(db, kb_id)

    # Get versions
    query = (
        select(KnowledgeBaseVersion)
        .where(KnowledgeBaseVersion.knowledge_base_id == kb_id)
        .order_by(KnowledgeBaseVersion.version_number.desc())
    )
    result = await db.execute(query)
    versions = result.scalars().all()

    return [
        KnowledgeBaseVersionResponse(
            id=v.id,
            knowledge_base_id=v.knowledge_base_id,
            version_number=v.version_number,
            change_type=v.change_type,
            document_snapshot=v.document_snapshot if isinstance(v.document_snapshot, list) else [],
            change_description=v.change_description,
            created_at=v.created_at,
        )
        for v in versions
    ]


@router.get(
    "/knowledge-bases/{kb_id}/status",
    response_model=dict[str, Any],
    summary="Get KB status",
    description="Get current status and statistics for a knowledge base.",
    responses={404: {"description": "Knowledge base not found"}},
)
async def get_kb_status(
    db: DbSession,
    kb_id: UUID,
) -> dict[str, Any]:
    """Get knowledge base status and statistics."""
    kb = await _get_kb_or_404(db, kb_id)

    # Calculate total size
    total_size = sum(doc.size_bytes or 0 for doc in kb.documents) if kb.documents else 0

    return {
        "id": str(kb.id),
        "status": kb.status,
        "current_version": kb.current_version,
        "document_count": len(kb.documents) if kb.documents else 0,
        "total_size_bytes": total_size,
        "storage_path": kb.storage_path,
        "index_path": kb.index_path,
    }


async def _perform_indexing_task(
    kb_id: UUID,
    rag_config_id: UUID | None,
    rag_adapter: RAGAdapterService,
) -> None:
    """Background task to perform indexing."""
    from app.database import get_db_context
    
    async with get_db_context() as db:
        try:
            # Reload KB with documents
            kb_query = select(KnowledgeBase).where(KnowledgeBase.id == kb_id).options(selectinload(KnowledgeBase.documents))
            kb_result = await db.execute(kb_query)
            kb = kb_result.scalar_one_or_none()
            
            if not kb:
                logger.error("KB not found in background task", kb_id=str(kb_id))
                return

            # Determine RAG configuration
            rag_config_model: RAGConfig | None = None
            if rag_config_id:
                config_query = select(RAGConfig).where(RAGConfig.id == rag_config_id)
                config_result = await db.execute(config_query)
                rag_config_model = config_result.scalar_one_or_none()

            if rag_config_model:
                rag = rag_adapter.get_or_create_rag(
                    rag_config_model, index_path=kb.index_path, force_new=True
                )
            else:
                # Fallback to default
                temp_config = RAGConfig(
                    project_id=kb.project_id,
                    name=f"Internal Indexing Config for {kb.name[:50]}",
                    rag_type="vector_semantic",
                    llm_provider="openai",
                    llm_model="gpt-4o-mini",
                    parameters={"collection_name": f"kb_{kb.id}"},
                )
                rag = rag_adapter.get_or_create_rag(
                    temp_config, index_path=kb.index_path, force_new=True
                )

            # Prepare documents
            await rag_adapter.prepare_documents(rag, kb.storage_path)

            # Create a new version
            await db.refresh(kb, ["documents"])
            await _create_version(
                db,
                kb,
                "reindexed",
                "Knowledge base re-indexed",
                documents=list(kb.documents) if kb.documents else [],
            )

            # Update document status to processed
            from sqlalchemy import update
            await db.execute(
                update(Document)
                .where(Document.knowledge_base_id == kb_id)
                .values(status="processed")
            )

            kb.status = "ready"
            await db.commit()
            logger.info("Successfully indexed knowledge base in background", kb_id=str(kb_id))

        except Exception as e:
            logger.exception("Failed to index knowledge base in background", kb_id=str(kb_id))
            # Critical: rollback before trying to update status to avoid PendingRollbackError
            await db.rollback()
            
            # Re-fetch KB in new transaction or after rollback to set error status
            kb_query = select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
            kb_result = await db.execute(kb_query)
            kb = kb_result.scalar_one_or_none()
            if kb:
                kb.status = "error"
                kb.metadata_["last_error"] = str(e)
                await db.commit()


@router.post(
    "/knowledge-bases/{kb_id}/index",
    response_model=KnowledgeBaseResponse,
    summary="Index a knowledge base",
    description="Trigger the indexing/document preparation process for a knowledge base.",
    responses={404: {"description": "Knowledge base not found"}},
)
async def index_knowledge_base(
    db: DbSession,
    kb_id: UUID,
    background_tasks: BackgroundTasks,
    request: KnowledgeBaseIndexRequest = KnowledgeBaseIndexRequest(),
    rag_adapter: RAGAdapterService = Depends(get_rag_adapter_service),
) -> KnowledgeBaseResponse:
    """Trigger indexing for a knowledge base."""
    kb = await _get_kb_or_404(db, kb_id)

    if not kb.documents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot index an empty knowledge base. Please upload documents first.",
        )

    if kb.status == "indexing":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Indexing is already in progress for this knowledge base.",
        )

    # Update status to indexing immediately
    kb.status = "indexing"
    await db.commit()
    await db.refresh(kb)

    # Add task to background
    background_tasks.add_task(
        _perform_indexing_task,
        kb_id=kb.id,
        rag_config_id=request.rag_config_id,
        rag_adapter=rag_adapter,
    )

    return _kb_to_response(kb)
