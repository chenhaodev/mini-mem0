"""Vector store operations using ChromaDB."""

import asyncio
import logging
from typing import List, Dict, Any, Optional

import chromadb

from settings import load_settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector storage and similarity search using ChromaDB."""

    def __init__(self, persist_directory: Optional[str] = None, ephemeral: bool = False):
        """
        Initialize ChromaDB client.

        Args:
            persist_directory: Directory for persistent storage (optional, loads from settings if not provided)
            ephemeral: If True, use in-memory storage (for testing)
        """
        if ephemeral:
            # Use ephemeral client for testing
            self.client = chromadb.EphemeralClient()
        else:
            if persist_directory is None:
                settings = load_settings()
                persist_directory = settings.chroma_persist_dir

            # Use persistent client for production
            self.client = chromadb.PersistentClient(path=persist_directory)

        # Single collection for all memories (patient_id in metadata for filtering)
        self.collection_name = "homecare_memories"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Patient memories for at-home care"}
        )

        logger.info(f"VectorStore initialized with collection: {self.collection_name}")

    async def add_embeddings(
        self,
        patient_id: str,
        memory_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """
        Add embedding to vector store.

        Args:
            patient_id: Patient identifier for filtering
            memory_id: Unique memory identifier
            embedding: Embedding vector (1536 dimensions for text-embedding-3-small)
            metadata: Additional metadata (category, etc.)
        """
        # GOTCHA: Chroma is synchronous, wrap in asyncio.to_thread
        await asyncio.to_thread(
            self._add_embedding_sync,
            patient_id,
            memory_id,
            embedding,
            metadata
        )

    def _add_embedding_sync(
        self,
        patient_id: str,
        memory_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """Synchronous add operation for Chroma."""
        # Add patient_id to metadata for filtering
        full_metadata = {
            "patient_id": patient_id,
            **metadata
        }

        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            metadatas=[full_metadata]
        )

        logger.debug(f"Added embedding for memory {memory_id} (patient: {patient_id})")

    async def search_similar(
        self,
        patient_id: str,
        query_embedding: List[float],
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search for similar memories using vector similarity.

        Args:
            patient_id: Patient identifier for filtering
            query_embedding: Query vector embedding
            limit: Maximum number of results (default 3, not 10 like mem0)

        Returns:
            List of similar memories with scores
        """
        # GOTCHA: Wrap in asyncio.to_thread
        results = await asyncio.to_thread(
            self._search_similar_sync,
            patient_id,
            query_embedding,
            limit
        )

        return results

    def _search_similar_sync(
        self,
        patient_id: str,
        query_embedding: List[float],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Synchronous search operation for Chroma."""
        # Filter by patient_id in metadata
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where={"patient_id": patient_id}
        )

        # Format results
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i, memory_id in enumerate(results['ids'][0]):
                # Calculate relevance score (Chroma returns distances, convert to similarity)
                # Lower distance = higher similarity
                distance = results['distances'][0][i] if results['distances'] else 0
                # Convert L2 distance to similarity score (0-1 range)
                relevance_score = max(0.0, 1.0 - (distance / 2.0))

                formatted_results.append({
                    "id": memory_id,
                    "score": relevance_score,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
                })

        logger.debug(f"Found {len(formatted_results)} similar memories for patient {patient_id}")
        return formatted_results

    async def delete_embedding(self, memory_id: str) -> None:
        """
        Delete embedding from vector store.

        Args:
            memory_id: Memory identifier to delete
        """
        # GOTCHA: Wrap in asyncio.to_thread
        await asyncio.to_thread(
            self._delete_embedding_sync,
            memory_id
        )

    def _delete_embedding_sync(self, memory_id: str) -> None:
        """Synchronous delete operation for Chroma."""
        try:
            self.collection.delete(ids=[memory_id])
            logger.debug(f"Deleted embedding for memory {memory_id}")
        except Exception as e:
            logger.warning(f"Failed to delete embedding {memory_id}: {e}")

    async def update_embedding(
        self,
        memory_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """
        Update existing embedding.

        Args:
            memory_id: Memory identifier
            embedding: New embedding vector
            metadata: Updated metadata
        """
        await asyncio.to_thread(
            self._update_embedding_sync,
            memory_id,
            embedding,
            metadata
        )

    def _update_embedding_sync(
        self,
        memory_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """Synchronous update operation for Chroma."""
        self.collection.update(
            ids=[memory_id],
            embeddings=[embedding],
            metadatas=[metadata]
        )
        logger.debug(f"Updated embedding for memory {memory_id}")


# Global vector store instance
vector_store = VectorStore()
