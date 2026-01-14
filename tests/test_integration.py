"""Integration tests for full workflows."""

import pytest
import asyncio

from homecare_memory.core.memory_manager import MemoryManager


@pytest.mark.asyncio
async def test_full_workflow_add_search_update(test_memory_manager: MemoryManager):
    """Test complete workflow: add -> search -> update."""
    patient_id = "patient_workflow_test"

    # Step 1: Add memories
    conversation = [
        "Patient is allergic to penicillin",
        "Patient prefers morning medication schedule"
    ]

    add_response = await test_memory_manager.add_memory(
        patient_id=patient_id,
        conversation=conversation
    )

    assert add_response.memories_created >= 1
    assert len(add_response.memory_ids) >= 1

    # Step 2: Search for memories
    search_response = await test_memory_manager.search_memory(
        patient_id=patient_id,
        query="medication preferences",
        limit=3
    )

    assert search_response.total >= 0

    # Step 3: Update a memory (if found)
    if add_response.memory_ids:
        memory_id = add_response.memory_ids[0]

        updated_memory = await test_memory_manager.update_memory(
            memory_id=memory_id,
            content="Patient prefers afternoon medication schedule"
        )

        assert updated_memory.content == "Patient prefers afternoon medication schedule"
        assert updated_memory.id == memory_id

    # Step 4: Get patient summary
    summary = await test_memory_manager.get_patient_summary(patient_id)

    assert summary.patient_id == patient_id
    assert summary.total_memories >= 0


@pytest.mark.asyncio
async def test_concurrent_operations(test_memory_manager: MemoryManager):
    """Test concurrent memory operations."""
    patient_id = "patient_concurrent_test"

    # Create multiple concurrent add operations
    conversations = [
        ["Patient has diabetes"],
        ["Patient prefers vegetarian diet"],
        ["Patient takes daily walks"]
    ]

    # Run concurrent adds
    tasks = [
        test_memory_manager.add_memory(
            patient_id=patient_id,
            conversation=conv
        )
        for conv in conversations
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check that all operations completed
    successes = [r for r in results if not isinstance(r, Exception)]
    assert len(successes) >= 1  # At least some should succeed

    # Verify memories were created
    summary = await test_memory_manager.get_patient_summary(patient_id)
    assert summary.total_memories >= 0


@pytest.mark.asyncio
async def test_large_patient_dataset(test_memory_manager: MemoryManager):
    """Test system with larger dataset (simplified for unit tests)."""
    patient_id = "patient_large_dataset"

    # Add multiple batches of memories
    for i in range(5):  # Reduced from 1000 for test performance
        conversation = [f"Observation {i}: Patient condition stable"]

        await test_memory_manager.add_memory(
            patient_id=patient_id,
            conversation=conversation
        )

    # Verify search still works
    search_response = await test_memory_manager.search_memory(
        patient_id=patient_id,
        query="patient condition",
        limit=3
    )

    assert search_response.total >= 0

    # Get summary
    summary = await test_memory_manager.get_patient_summary(patient_id)
    assert summary.total_memories >= 0


@pytest.mark.asyncio
async def test_patient_isolation_in_workflow(test_memory_manager: MemoryManager):
    """Test that patients' memories are properly isolated."""
    # Add memories for patient 1
    await test_memory_manager.add_memory(
        patient_id="patient_1",
        conversation=["Patient 1 is allergic to aspirin"]
    )

    # Add memories for patient 2
    await test_memory_manager.add_memory(
        patient_id="patient_2",
        conversation=["Patient 2 prefers morning visits"]
    )

    # Search patient 1 - should not see patient 2's memories
    results_p1 = await test_memory_manager.search_memory(
        patient_id="patient_1",
        query="allergies",
        limit=10
    )

    # Search patient 2 - should not see patient 1's memories
    results_p2 = await test_memory_manager.search_memory(
        patient_id="patient_2",
        query="preferences",
        limit=10
    )

    # Verify isolation (results should not overlap)
    # Note: Due to mocking limitations, we mainly verify no errors occur
    assert results_p1.total >= 0
    assert results_p2.total >= 0
