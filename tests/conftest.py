"""Pytest configuration and fixtures."""

import pytest
import asyncio
import os
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

from httpx import AsyncClient

from homecare_memory.main import app
from homecare_memory.db.pool import DatabasePool
from homecare_memory.core.vector_store import VectorStore
from homecare_memory.core.extractor import MemoryExtractor
from homecare_memory.core.memory_manager import MemoryManager


# Set test environment
os.environ["DATABASE_URL"] = "postgresql://homecare:dev123@localhost:5432/homecare_test_db"
os.environ["OPENAI_API_KEY"] = "sk-test-key"
os.environ["CHROMA_PERSIST_DIR"] = "./test_chroma_db"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def test_db() -> AsyncGenerator[DatabasePool, None]:
    """
    Provide test database with transaction rollback.

    Creates a database pool and rolls back all changes after each test.
    """
    # Create test database pool
    db_pool = DatabasePool(os.environ["DATABASE_URL"])
    await db_pool.initialize()

    # Create schema (skip if already exists)
    async with db_pool.acquire() as conn:
        # Check if table exists
        table_exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'memories'
            )
            """
        )

        if not table_exists:
            # Load schema from migration file
            schema_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "db",
                "migrations",
                "001_init.sql"
            )
            with open(schema_path, "r") as f:
                schema_sql = f.read()
            await conn.execute(schema_sql)

    yield db_pool

    # Cleanup
    async with db_pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS memories CASCADE")

    await db_pool.close()


@pytest.fixture
async def test_vector_store() -> AsyncGenerator[VectorStore, None]:
    """Provide test vector store with in-memory ChromaDB."""
    # Use test directory for Chroma
    vector_store = VectorStore(persist_directory="./test_chroma_db")

    yield vector_store

    # Cleanup: Delete test collection
    try:
        vector_store.client.delete_collection(vector_store.collection_name)
    except Exception:
        pass


@pytest.fixture
def mock_openai_extractor() -> MemoryExtractor:
    """Provide mock memory extractor."""
    extractor = MemoryExtractor(api_key="sk-test-key")

    # Mock the OpenAI client
    extractor.client = AsyncMock()

    # Mock extraction response
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
                    "metadata": {}
                },
                {
                    "category": "preference",
                    "priority": "normal",
                    "content": "Patient prefers morning walks",
                    "metadata": {}
                }
            ]
        }
        '''
        return mock_response

    extractor.client.chat.completions.create = mock_create

    return extractor


@pytest.fixture
async def test_memory_manager(
    test_db: DatabasePool,
    test_vector_store: VectorStore,
    mock_openai_extractor: MemoryExtractor
) -> AsyncGenerator[MemoryManager, None]:
    """Provide test memory manager with mocked dependencies."""
    manager = MemoryManager(
        db_pool=test_db,
        vector_store=test_vector_store,
        extractor=mock_openai_extractor,
        openai_api_key="sk-test-key"
    )

    # Mock OpenAI embeddings
    async def mock_embeddings(*args, **kwargs):
        mock_response = MagicMock()
        # Return dummy embeddings (1536 dimensions)
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        return mock_response

    manager.openai_client.embeddings.create = mock_embeddings

    yield manager


@pytest.fixture
async def test_client() -> AsyncGenerator[AsyncClient, None]:
    """Provide test HTTP client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
