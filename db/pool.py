"""Database connection pool management for PostgreSQL."""

import asyncpg
from asyncpg.pool import Pool
from contextlib import asynccontextmanager
from typing import Optional
import logging

from homecare_memory.settings import load_settings

logger = logging.getLogger(__name__)


class DatabasePool:
    """Manages PostgreSQL connection pool."""

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database pool.

        Args:
            database_url: PostgreSQL connection URL (optional, loads from settings if not provided)
        """
        if database_url:
            self.database_url = database_url
        else:
            settings = load_settings()
            self.database_url = settings.database_url
            self.min_size = settings.db_pool_min_size
            self.max_size = settings.db_pool_max_size

        self.pool: Optional[Pool] = None

    async def initialize(self):
        """Create connection pool."""
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=getattr(self, 'min_size', 5),
                max_size=getattr(self, 'max_size', 20),
                max_inactive_connection_lifetime=300,
                command_timeout=60
            )
            logger.info("Database connection pool initialized")

    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Database connection pool closed")

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        if not self.pool:
            await self.initialize()

        async with self.pool.acquire() as connection:
            yield connection


# Global database pool instance
db_pool = DatabasePool()


async def initialize_database():
    """Initialize database connection pool."""
    await db_pool.initialize()


async def close_database():
    """Close database connection pool."""
    await db_pool.close()


async def test_connection() -> bool:
    """
    Test database connection.

    Returns:
        True if connection successful
    """
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        logger.info("Database connection test successful")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False
