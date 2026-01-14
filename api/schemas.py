"""API request/response schemas."""

from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import UUID

from core.models import Memory, MemorySearchResult, MemoryCategory


class AddMemoryRequest(BaseModel):
    """Request to add memory from conversation."""

    patient_id: str
    conversation: List[str] = Field(..., min_items=1)


class AddMemoryResponse(BaseModel):
    """Response with created memories."""

    memories_created: int
    memory_ids: List[UUID]


class SearchMemoryRequest(BaseModel):
    """Search memories by semantic query."""

    patient_id: str
    query: str
    limit: int = Field(default=3, ge=1, le=10)
    category_filter: Optional[MemoryCategory] = None


class SearchMemoryResponse(BaseModel):
    """Search results with relevance scores."""

    results: List[MemorySearchResult]
    total: int


class UpdateMemoryRequest(BaseModel):
    """Request to update memory content."""

    content: str = Field(..., min_length=1, max_length=2000)


class PatientSummaryResponse(BaseModel):
    """Patient summary with categorized memories."""

    patient_id: str
    total_memories: int
    critical_memories: int
    memories_by_category: dict[str, int]
    recent_observations: List[Memory]
