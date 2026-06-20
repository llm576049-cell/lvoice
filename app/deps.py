import asyncio

from fastapi import HTTPException

from app.config import settings
from app.engines.base import BaseTTSEngine

_engine: BaseTTSEngine | None = None
_inference_semaphore = asyncio.Semaphore(settings.max_concurrent_requests)


def set_engine(engine: BaseTTSEngine) -> None:
    global _engine
    _engine = engine


def get_engine() -> BaseTTSEngine:
    if _engine is None:
        raise RuntimeError("Engine not initialized")
    return _engine


def require_engine() -> BaseTTSEngine:
    """FastAPI dependency: like `get_engine`, but reports a 503 instead of a bare
    RuntimeError when the engine hasn't finished loading."""
    try:
        return get_engine()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail="Engine not ready") from e


def get_inference_semaphore() -> asyncio.Semaphore:
    return _inference_semaphore
