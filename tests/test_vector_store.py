"""Tests for VectorStore."""

import pytest

from core.vector_store import VectorStore


@pytest.mark.asyncio
async def test_add_and_search_embeddings(test_vector_store: VectorStore):
    """Add and search embeddings."""
    # Add embedding
    test_embedding = [0.1] * 1536  # 1536 dimensions for text-embedding-3-small
    await test_vector_store.add_embeddings(
        patient_id="patient_123",
        memory_id="memory_001",
        embedding=test_embedding,
        metadata={"category": "allergy"}
    )

    # Search similar
    query_embedding = [0.1] * 1536  # Same embedding should match
    results = await test_vector_store.search_similar(
        patient_id="patient_123",
        query_embedding=query_embedding,
        limit=3
    )

    assert len(results) >= 1
    assert results[0]["id"] == "memory_001"
    assert results[0]["score"] > 0.5  # Should have high similarity


@pytest.mark.asyncio
async def test_patient_isolation(test_vector_store: VectorStore):
    """Memories are isolated by patient_id."""
    # Add memory for patient 1
    embedding1 = [0.1] * 1536
    await test_vector_store.add_embeddings(
        patient_id="patient_1",
        memory_id="memory_p1",
        embedding=embedding1,
        metadata={"category": "allergy"}
    )

    # Add memory for patient 2
    embedding2 = [0.2] * 1536
    await test_vector_store.add_embeddings(
        patient_id="patient_2",
        memory_id="memory_p2",
        embedding=embedding2,
        metadata={"category": "allergy"}
    )

    # Search for patient 1 - should only return patient 1's memories
    results = await test_vector_store.search_similar(
        patient_id="patient_1",
        query_embedding=[0.1] * 1536,
        limit=10
    )

    # All results should be for patient_1
    for result in results:
        assert result["metadata"]["patient_id"] == "patient_1"


@pytest.mark.asyncio
async def test_delete_embedding(test_vector_store: VectorStore):
    """Delete embedding from vector store."""
    # Add embedding
    test_embedding = [0.1] * 1536
    await test_vector_store.add_embeddings(
        patient_id="patient_123",
        memory_id="memory_delete_test",
        embedding=test_embedding,
        metadata={"category": "preference"}
    )

    # Verify it exists
    results_before = await test_vector_store.search_similar(
        patient_id="patient_123",
        query_embedding=test_embedding,
        limit=10
    )

    assert any(r["id"] == "memory_delete_test" for r in results_before)

    # Delete it
    await test_vector_store.delete_embedding("memory_delete_test")

    # Verify it's gone
    results_after = await test_vector_store.search_similar(
        patient_id="patient_123",
        query_embedding=test_embedding,
        limit=10
    )

    assert not any(r["id"] == "memory_delete_test" for r in results_after)


@pytest.mark.asyncio
async def test_update_embedding(test_vector_store: VectorStore):
    """Update existing embedding."""
    # Add initial embedding
    initial_embedding = [0.1] * 1536
    await test_vector_store.add_embeddings(
        patient_id="patient_123",
        memory_id="memory_update_test",
        embedding=initial_embedding,
        metadata={"category": "preference"}
    )

    # Update with new embedding
    new_embedding = [0.9] * 1536
    await test_vector_store.update_embedding(
        memory_id="memory_update_test",
        embedding=new_embedding,
        metadata={"category": "preference", "patient_id": "patient_123"}
    )

    # Search with new embedding should find it
    results = await test_vector_store.search_similar(
        patient_id="patient_123",
        query_embedding=new_embedding,
        limit=3
    )

    assert len(results) >= 1
    assert any(r["id"] == "memory_update_test" for r in results)
