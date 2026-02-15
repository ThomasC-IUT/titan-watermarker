"""
Document Fortress — Main Application Entry Point
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import config
from app.api.routes import router as api_router


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s │ %(name)-22s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fortress")


# ──────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("=" * 60)
    logger.info("  TITAN VISUAL WATERMARKER v2.2 — Starting Up")
    logger.info("=" * 60)

    # Ensure directories
    config.ensure_dirs()
    logger.info("Data directories ready: %s", config.DATA_DIR)

    logger.info("=" * 60)
    logger.info("  Server ready — http://0.0.0.0:8000")
    logger.info("=" * 60)

    yield  # App is running

    logger.info("Titan shutting down.")


# ──────────────────────────────────────────────
# App Creation
# ──────────────────────────────────────────────
app = FastAPI(
    title="Titan Visual Watermarker",
    description="High-performance, visual-only watermarking service for PDFs and Images.",
    version="2.2.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(api_router)

# Static files (frontend)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def root():
    """Serve the frontend."""
    return FileResponse("app/static/index.html")
