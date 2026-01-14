"""Tests for MemoryExtractor."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from core.extractor import MemoryExtractor
from core.models import MemoryCategory, Priority


@pytest.mark.asyncio
async def test_extract_memories_from_conversation():
    """Extract memories from conversation using mocked OpenAI."""
    extractor = MemoryExtractor(api_key="sk-test-key")

    # Mock OpenAI response
    extractor.client = AsyncMock()

    async def mock_create(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.function_call = MagicMock()
        mock_response.choices[0].message.function_call.arguments = '''
        {
            "memories": [
                {
                    "category": "allergy",
                    "priority": "critical",
                    "content": "Patient is allergic to penicillin",
                    "metadata": {"severity": "high"}
                },
                {
                    "category": "preference",
                    "priority": "normal",
                    "content": "Patient prefers morning medication",
                    "metadata": {}
                }
            ]
        }
        '''
        return mock_response

    extractor.client.chat.completions.create = mock_create

    # Extract memories
    conversation = [
        "Patient mentioned they are allergic to penicillin",
        "They prefer taking medication in the morning"
    ]

    memories = await extractor.extract_memories(conversation)

    assert len(memories) == 2

    # Check first memory (allergy)
    assert memories[0].category == MemoryCategory.ALLERGY
    assert memories[0].priority == Priority.CRITICAL
    assert "penicillin" in memories[0].content.lower()

    # Check second memory (preference)
    assert memories[1].category == MemoryCategory.PREFERENCE
    assert memories[1].priority == Priority.NORMAL


@pytest.mark.asyncio
async def test_extract_memories_empty_conversation():
    """Handle empty extraction gracefully."""
    extractor = MemoryExtractor(api_key="sk-test-key")

    # Mock OpenAI response with no memories
    extractor.client = AsyncMock()

    async def mock_create(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.function_call = MagicMock()
        mock_response.choices[0].message.function_call.arguments = '{"memories": []}'
        return mock_response

    extractor.client.chat.completions.create = mock_create

    memories = await extractor.extract_memories(["Nothing relevant here"])

    assert len(memories) == 0


@pytest.mark.asyncio
async def test_extract_memories_api_failure():
    """Handle API failures gracefully."""
    extractor = MemoryExtractor(api_key="sk-test-key")

    # Mock API failure
    extractor.client = AsyncMock()
    extractor.client.chat.completions.create = AsyncMock(
        side_effect=Exception("API Error")
    )

    with pytest.raises(Exception) as exc_info:
        await extractor.extract_memories(["Some conversation"])

    assert "API Error" in str(exc_info.value) or "Memory extraction failed" in str(exc_info.value)
