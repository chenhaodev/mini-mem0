"""FastAPI routes for memory management."""

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends

from homecare_memory.api.schemas import (
    AddMemoryRequest,
    AddMemoryResponse,
    SearchMemoryRequest,
    SearchMemoryResponse,
    UpdateMemoryRequest,
    PatientSummaryResponse
)
from homecare_memory.core.models import Memory
from homecare_memory.core.memory_manager import MemoryManager
from homecare_memory.core.extractor import memory_extractor
from homecare_memory.core.vector_store import vector_store
from homecare_memory.db.pool import db_pool

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["memories"])


# Dependency injection for MemoryManager
async def get_memory_manager() -> MemoryManager:
    """Get memory manager instance."""
    return MemoryManager(
        db_pool=db_pool,
        vector_store=vector_store,
        extractor=memory_extractor
    )


@router.post("/memories", response_model=AddMemoryResponse, status_code=201)
async def add_memory(
    request: AddMemoryRequest,
    manager: MemoryManager = Depends(get_memory_manager)
) -> AddMemoryResponse:
    """
    Add memories from conversation.

    Extracts structured memories from conversation using LLM,
    generates embeddings, and stores in database + vector store.

    - **patient_id**: Unique patient identifier
    - **conversation**: List of conversation messages to extract memories from
    """
    try:
        response = await manager.add_memory(
            patient_id=request.patient_id,
            conversation=request.conversation
        )
        return response
    except Exception as e:
        logger.error(f"Failed to add memories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add memories: {str(e)}")


@router.post("/memories/search", response_model=SearchMemoryResponse)
async def search_memories(
    request: SearchMemoryRequest,
    manager: MemoryManager = Depends(get_memory_manager)
) -> SearchMemoryResponse:
    """
    Search memories by semantic query.

    Performs vector similarity search and returns relevant memories
    sorted by priority (CRITICAL first) and relevance.

    Target latency: <100ms

    - **patient_id**: Patient identifier
    - **query**: Search query
    - **limit**: Maximum results (default: 3)
    - **category_filter**: Optional category filter
    """
    try:
        response = await manager.search_memory(
            patient_id=request.patient_id,
            query=request.query,
            limit=request.limit,
            category_filter=request.category_filter
        )
        return response
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.patch("/memories/{memory_id}", response_model=Memory)
async def update_memory(
    memory_id: UUID,
    request: UpdateMemoryRequest,
    manager: MemoryManager = Depends(get_memory_manager)
) -> Memory:
    """
    Update existing memory content.

    Updates memory content and regenerates embedding.
    Uses simple timestamp-based versioning (no LLM-based conflict resolution).

    - **memory_id**: Memory identifier to update
    - **content**: New memory content
    """
    try:
        memory = await manager.update_memory(
            memory_id=memory_id,
            content=request.content
        )
        return memory
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


@router.get("/patients/{patient_id}/summary", response_model=PatientSummaryResponse)
async def get_patient_summary(
    patient_id: str,
    manager: MemoryManager = Depends(get_memory_manager)
) -> PatientSummaryResponse:
    """
    Get comprehensive patient summary.

    Returns aggregate view of all active memories:
    - Total memory count
    - Critical memories count
    - Memories grouped by category
    - Recent observations (last 30 days)

    - **patient_id**: Patient identifier
    """
    try:
        summary = await manager.get_patient_summary(patient_id)
        return summary
    except Exception as e:
        logger.error(f"Failed to get summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "homecare-memory",
        "version": "1.0.0"
    }
