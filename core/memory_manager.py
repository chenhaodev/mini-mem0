"""Core memory management business logic."""

import logging
import json
from typing import List, Optional
from uuid import UUID
from datetime import datetime, timedelta

from openai import AsyncOpenAI

from core.models import (
    Memory,
    ExtractedMemory,
    MemoryCategory,
    Priority,
    MemorySearchResult
)
from core.extractor import MemoryExtractor
from core.vector_store import VectorStore
from db.pool import DatabasePool
from settings import load_settings
from api.schemas import (
    AddMemoryResponse,
    SearchMemoryResponse,
    PatientSummaryResponse
)

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages memory operations with optimized conflict handling."""

    def __init__(
        self,
        db_pool: DatabasePool,
        vector_store: VectorStore,
        extractor: MemoryExtractor,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize memory manager.

        Args:
            db_pool: Database connection pool
            vector_store: Vector store for embeddings
            extractor: Memory extractor
            openai_api_key: OpenAI API key for embeddings
        """
        self.db_pool = db_pool
        self.vector_store = vector_store
        self.extractor = extractor

        # OpenAI client for embeddings
        if openai_api_key is None:
            settings = load_settings()
            openai_api_key = settings.openai_api_key
            self.embedding_model = settings.embedding_model
        else:
            self.embedding_model = "text-embedding-3-small"

        self.openai_client = AsyncOpenAI(api_key=openai_api_key)

    async def add_memory(
        self,
        patient_id: str,
        conversation: List[str]
    ) -> AddMemoryResponse:
        """
        Add memories from conversation with optimized conflict handling.

        CRITICAL OPTIMIZATION:
        - Critical memories (allergies, medications): Check contradictions
        - Normal memories (preferences, observations): Fast-path direct insert

        Args:
            patient_id: Patient identifier
            conversation: List of conversation messages

        Returns:
            Response with created memory IDs

        Raises:
            Exception: If memory creation fails
        """
        try:
            # STEP 1: Extract structured memories using LLM (only LLM use in write path)
            extracted = await self.extractor.extract_memories(conversation)

            if not extracted:
                logger.warning("No memories extracted from conversation")
                return AddMemoryResponse(memories_created=0, memory_ids=[])

            # STEP 2: Generate embeddings in batch (not one-by-one)
            embeddings = await self._batch_generate_embeddings(
                [mem.content for mem in extracted]
            )

            created_ids = []

            for memory_data, embedding in zip(extracted, embeddings):
                # STEP 3: Fast-path vs validation-path
                if memory_data.priority == Priority.CRITICAL:
                    # CRITICAL PATH: Simple contradiction check
                    # GOTCHA: Use string similarity, NOT LLM comparison (speed)
                    existing = await self.search_memory(
                        patient_id,
                        memory_data.content,
                        limit=2,  # Only check top 2, not 10
                        category_filter=memory_data.category
                    )

                    # Simple rule: If >90% similar and contradicts, update instead of add
                    if existing.results and existing.results[0].relevance_score > 0.9:
                        # Check for contradiction using simple logic
                        if self._is_contradiction(memory_data, existing.results[0].memory):
                            await self.update_memory(
                                existing.results[0].memory.id,
                                memory_data.content
                            )
                            created_ids.append(existing.results[0].memory.id)
                            continue

                # STEP 4: Insert into PostgreSQL + Chroma
                # PATTERN: Use transaction for atomicity
                async with self.db_pool.acquire() as conn:
                    async with conn.transaction():
                        # Insert to PostgreSQL
                        memory_id = await conn.fetchval(
                            """
                            INSERT INTO memories
                            (patient_id, category, priority, content, metadata, created_at, updated_at)
                            VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
                            RETURNING id
                            """,
                            patient_id,
                            memory_data.category.value,
                            memory_data.priority.value,
                            memory_data.content,
                            json.dumps(memory_data.metadata)
                        )

                        # Add to vector store
                        await self.vector_store.add_embeddings(
                            patient_id=patient_id,
                            memory_id=str(memory_id),
                            embedding=embedding,
                            metadata={"category": memory_data.category.value}
                        )

                        created_ids.append(memory_id)

            logger.info(f"Created {len(created_ids)} memories for patient {patient_id}")
            return AddMemoryResponse(
                memories_created=len(created_ids),
                memory_ids=created_ids
            )

        except Exception as e:
            logger.error(f"Failed to add memories: {e}")
            raise

    def _is_contradiction(self, new: ExtractedMemory, existing: Memory) -> bool:
        """
        Simple rule-based contradiction detection.
        NOT using LLM (speed optimization).

        Args:
            new: New extracted memory
            existing: Existing memory from database

        Returns:
            True if contradiction detected
        """
        new_content_lower = new.content.lower()
        existing_content_lower = existing.content.lower()

        # PATTERN: Category-specific rules
        if new.category == MemoryCategory.ALLERGY:
            # If existing says "no allergy to X" and new says "allergic to X"
            # Simple keyword matching suffices for this domain
            if "no allergy" in existing_content_lower and "allergic" in new_content_lower:
                return True
            if "not allergic" in existing_content_lower and "allergic to" in new_content_lower:
                return True

        if new.category == MemoryCategory.MEDICATION:
            # Check for dosage contradictions by looking for different numbers
            # This is a simplified check - could be enhanced
            import re
            new_numbers = set(re.findall(r'\d+', new_content_lower))
            existing_numbers = set(re.findall(r'\d+', existing_content_lower))

            # If same medication but different dosages
            if new_numbers and existing_numbers and new_numbers != existing_numbers:
                return True

        return False

    async def search_memory(
        self,
        patient_id: str,
        query: str,
        limit: int = 3,  # Default 3, not 10 like mem0
        category_filter: Optional[MemoryCategory] = None
    ) -> SearchMemoryResponse:
        """
        Fast semantic search optimized for at-home care.
        TARGET: <100ms latency

        Args:
            patient_id: Patient identifier
            query: Search query
            limit: Maximum results to return
            category_filter: Optional category filter

        Returns:
            Search results with relevance scores
        """
        try:
            # STEP 1: Generate query embedding
            query_embedding = await self._generate_embedding(query)

            # STEP 2: Vector search (limit=3 for speed)
            similar = await self.vector_store.search_similar(
                patient_id=patient_id,
                query_embedding=query_embedding,
                limit=limit
            )

            if not similar:
                return SearchMemoryResponse(results=[], total=0)

            # STEP 3: Fetch full records from PostgreSQL
            memory_ids = [result["id"] for result in similar]

            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, patient_id, category, priority, content,
                           metadata, created_at, updated_at
                    FROM memories
                    WHERE id = ANY($1::uuid[])
                      AND deleted_at IS NULL
                      AND ($2::text IS NULL OR category = $2)
                    ORDER BY
                        CASE priority
                            WHEN 'critical' THEN 1
                            WHEN 'high' THEN 2
                            ELSE 3
                        END,
                        created_at DESC
                    """,
                    memory_ids,
                    category_filter.value if category_filter else None
                )

            # STEP 4: Combine with relevance scores
            # Create a map of memory_id to score
            score_map = {result["id"]: result["score"] for result in similar}

            results = []
            for row in rows:
                memory_id = str(row["id"])
                relevance_score = score_map.get(memory_id, 0.0)

                memory = Memory(
                    id=row["id"],
                    patient_id=row["patient_id"],
                    category=MemoryCategory(row["category"]),
                    priority=Priority(row["priority"]),
                    content=row["content"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    deleted_at=None
                )

                results.append(MemorySearchResult(
                    memory=memory,
                    relevance_score=relevance_score
                ))

            logger.info(f"Found {len(results)} memories for query (patient: {patient_id})")
            return SearchMemoryResponse(results=results, total=len(results))

        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            raise

    async def update_memory(
        self,
        memory_id: UUID,
        content: str
    ) -> Memory:
        """
        Update existing memory content.
        SIMPLE: Direct update with new timestamp.
        No LLM-based conflict resolution.

        Args:
            memory_id: Memory identifier
            content: New content

        Returns:
            Updated memory

        Raises:
            ValueError: If memory not found
        """
        try:
            # Generate new embedding
            embedding = await self._generate_embedding(content)

            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    # Update in PostgreSQL
                    row = await conn.fetchrow(
                        """
                        UPDATE memories
                        SET content = $1, updated_at = NOW()
                        WHERE id = $2 AND deleted_at IS NULL
                        RETURNING id, patient_id, category, priority, content,
                                  metadata, created_at, updated_at
                        """,
                        content,
                        memory_id
                    )

                    if not row:
                        raise ValueError(f"Memory {memory_id} not found")

                    # Update in vector store
                    await self.vector_store.update_embedding(
                        memory_id=str(memory_id),
                        embedding=embedding,
                        metadata={"category": row["category"]}
                    )

            memory = Memory(
                id=row["id"],
                patient_id=row["patient_id"],
                category=MemoryCategory(row["category"]),
                priority=Priority(row["priority"]),
                content=row["content"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                deleted_at=None
            )

            logger.info(f"Updated memory {memory_id}")
            return memory

        except Exception as e:
            logger.error(f"Memory update failed: {e}")
            raise

    async def get_patient_summary(self, patient_id: str) -> PatientSummaryResponse:
        """
        Get comprehensive patient summary.
        Aggregate view of all active memories.
        Group by category, prioritize CRITICAL.
        Include recent observations (last 30 days).

        Args:
            patient_id: Patient identifier

        Returns:
            Patient summary with categorized memories
        """
        try:
            async with self.db_pool.acquire() as conn:
                # Get total counts
                total_row = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE priority = 'critical') as critical_count
                    FROM memories
                    WHERE patient_id = $1 AND deleted_at IS NULL
                    """,
                    patient_id
                )

                # Get counts by category
                category_rows = await conn.fetch(
                    """
                    SELECT category, COUNT(*) as count
                    FROM memories
                    WHERE patient_id = $1 AND deleted_at IS NULL
                    GROUP BY category
                    """,
                    patient_id
                )

                # Get recent observations (last 30 days)
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                recent_rows = await conn.fetch(
                    """
                    SELECT id, patient_id, category, priority, content,
                           metadata, created_at, updated_at
                    FROM memories
                    WHERE patient_id = $1
                      AND deleted_at IS NULL
                      AND category = 'observation'
                      AND created_at >= $2
                    ORDER BY created_at DESC
                    LIMIT 10
                    """,
                    patient_id,
                    cutoff_date
                )

            # Format response
            memories_by_category = {row["category"]: row["count"] for row in category_rows}

            recent_observations = [
                Memory(
                    id=row["id"],
                    patient_id=row["patient_id"],
                    category=MemoryCategory(row["category"]),
                    priority=Priority(row["priority"]),
                    content=row["content"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    deleted_at=None
                )
                for row in recent_rows
            ]

            return PatientSummaryResponse(
                patient_id=patient_id,
                total_memories=total_row["total"],
                critical_memories=total_row["critical_count"],
                memories_by_category=memories_by_category,
                recent_observations=recent_observations
            )

        except Exception as e:
            logger.error(f"Failed to get patient summary: {e}")
            raise

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        response = await self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    async def _batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings in batch for efficiency.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Batch generate embeddings (OpenAI allows up to 2048 texts per request)
        response = await self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )

        return [item.embedding for item in response.data]
