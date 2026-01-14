"""Pydantic domain models for Homecare Memory."""

from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4


class Priority(str, Enum):
    """Memory priority levels."""

    CRITICAL = "critical"  # Allergies, critical medications
    HIGH = "high"  # Medications, medical conditions
    NORMAL = "normal"  # Preferences, observations


class MemoryCategory(str, Enum):
    """Memory category types for at-home care."""

    MEDICAL_HISTORY = "medical_history"  # Conditions, diagnoses
    ALLERGY = "allergy"  # Critical: allergies
    MEDICATION = "medication"  # Critical: current medications
    PREFERENCE = "preference"  # Dietary, comfort preferences
    OBSERVATION = "observation"  # Caregiver notes
    APPOINTMENT = "appointment"  # Medical appointments


class Memory(BaseModel):
    """Core memory model."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4)
    patient_id: str = Field(..., description="Unique patient identifier")
    category: MemoryCategory
    priority: Priority
    content: str = Field(..., min_length=1, max_length=2000)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    deleted_at: Optional[datetime] = None


class MemorySearchResult(BaseModel):
    """Search result with relevance score."""

    memory: Memory
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class ExtractedMemory(BaseModel):
    """Memory extracted from conversation by LLM."""

    category: MemoryCategory
    priority: Priority
    content: str = Field(..., min_length=1, max_length=2000)
    metadata: Dict[str, Any] = Field(default_factory=dict)
