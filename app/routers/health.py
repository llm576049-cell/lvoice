from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app import deps

router = APIRouter()


@router.get("/healthz")
def healthz():
    return {"status": "ok"}


@router.get("/readyz")
def readyz():
    try:
        deps.get_engine()
    except RuntimeError:
        return JSONResponse({"status": "not_ready"}, status_code=503)
    return {"status": "ready"}
