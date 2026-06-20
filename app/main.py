import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import deps
from app.config import settings
from app.engines.cosyvoice import build_engine
from app.routers import health, tts, voices

logger = logging.getLogger("lvoice")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        deps.set_engine(build_engine())
        logger.info("Engine '%s' loaded", settings.engine)
    except Exception:
        logger.exception("Failed to load engine '%s'; running without it", settings.engine)
    yield


app = FastAPI(title="lvoice", lifespan=lifespan)
app.include_router(health.router)
app.include_router(tts.router)
app.include_router(voices.router)
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
