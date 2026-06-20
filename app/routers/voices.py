from fastapi import APIRouter, Depends

from app.deps import require_engine
from app.engines.base import BaseTTSEngine
from app.schemas import VoicesResponse

router = APIRouter(prefix="/v1", tags=["voices"])


@router.get("/voices", response_model=VoicesResponse)
def list_voices(engine: BaseTTSEngine = Depends(require_engine)):
    return VoicesResponse(voices=engine.list_voices())
