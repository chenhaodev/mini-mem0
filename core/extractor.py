"""Memory extraction from conversations using OpenAI function calling."""

import logging
from typing import List
import json

from openai import AsyncOpenAI

from homecare_memory.core.models import ExtractedMemory, MemoryCategory, Priority
from homecare_memory.settings import load_settings

logger = logging.getLogger(__name__)


class MemoryExtractor:
    """Extracts structured memories from conversations using LLM function calling."""

    def __init__(self, api_key: str = None):
        """
        Initialize memory extractor.

        Args:
            api_key: OpenAI API key (optional, loads from settings if not provided)
        """
        if api_key is None:
            settings = load_settings()
            api_key = settings.openai_api_key

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"  # Fast and cost-effective for extraction

        # Function schema for structured memory extraction
        self.extraction_function = {
            "name": "extract_memories",
            "description": "Extract patient memories from conversation for at-home care",
            "parameters": {
                "type": "object",
                "properties": {
                    "memories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {
                                    "type": "string",
                                    "enum": [
                                        "medical_history",
                                        "allergy",
                                        "medication",
                                        "preference",
                                        "observation",
                                        "appointment"
                                    ],
                                    "description": "Memory category"
                                },
                                "priority": {
                                    "type": "string",
                                    "enum": ["critical", "high", "normal"],
                                    "description": "Priority level (critical for allergies/medications, normal for preferences)"
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Clear, concise memory content (1-2000 characters)"
                                },
                                "metadata": {
                                    "type": "object",
                                    "description": "Additional context (dosage, frequency, etc.)",
                                    "additionalProperties": True
                                }
                            },
                            "required": ["category", "priority", "content"]
                        }
                    }
                },
                "required": ["memories"]
            }
        }

    async def extract_memories(
        self,
        conversation: List[str]
    ) -> List[ExtractedMemory]:
        """
        Extract structured memories from conversation.

        Args:
            conversation: List of conversation messages

        Returns:
            List of extracted memories with category, priority, content

        Raises:
            Exception: If extraction fails
        """
        try:
            # PATTERN: Batch all messages to single LLM call for efficiency
            conversation_text = "\n".join([
                f"Message {i+1}: {msg}"
                for i, msg in enumerate(conversation)
            ])

            # System prompt for memory extraction
            system_prompt = """You are a medical memory extraction assistant for at-home care.
Extract patient information from conversations into structured memories.

Guidelines:
- ALLERGIES and MEDICATIONS are CRITICAL priority
- Medical conditions are HIGH priority
- Preferences and observations are NORMAL priority
- Extract clear, factual statements only
- Include relevant metadata (dosage, frequency, dates)
- Avoid duplicates or vague statements
"""

            # Call OpenAI with function calling
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": conversation_text}
                ],
                functions=[self.extraction_function],
                function_call={"name": "extract_memories"},
                max_tokens=2000,  # GOTCHA: Prevent truncation
                temperature=0.0  # Deterministic extraction
            )

            # Parse function call response
            message = response.choices[0].message

            if not message.function_call:
                logger.warning("No memories extracted from conversation")
                return []

            # Parse extracted memories
            function_args = json.loads(message.function_call.arguments)
            memories_data = function_args.get("memories", [])

            # Convert to ExtractedMemory objects
            extracted_memories = []
            for mem_data in memories_data:
                try:
                    memory = ExtractedMemory(
                        category=MemoryCategory(mem_data["category"]),
                        priority=Priority(mem_data["priority"]),
                        content=mem_data["content"],
                        metadata=mem_data.get("metadata", {})
                    )
                    extracted_memories.append(memory)
                except Exception as e:
                    logger.warning(f"Failed to parse extracted memory: {e}")
                    continue

            logger.info(f"Extracted {len(extracted_memories)} memories from conversation")
            return extracted_memories

        except Exception as e:
            logger.error(f"Memory extraction failed: {e}")
            raise


# Global extractor instance
memory_extractor = MemoryExtractor()
