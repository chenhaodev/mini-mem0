"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from db.pool import initialize_database, close_database
from settings import load_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown events.

    Startup:
    - Initialize database connection pool
    - Verify database connectivity
    - Initialize vector store

    Shutdown:
    - Close database connection pool
    """
    # Startup
    logger.info("Starting Homecare Memory service...")

    try:
        # Load settings to verify configuration
        load_settings()
        logger.info("Settings loaded successfully")

        # Initialize database pool
        await initialize_database()
        logger.info("Database pool initialized")

        logger.info("Homecare Memory service started successfully")

    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Homecare Memory service...")
    await close_database()
    logger.info("Service shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Homecare Memory API",
    description="Simplified memory management system for at-home care",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (configure as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Homecare Memory API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
