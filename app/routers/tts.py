from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from app.audio import wav_bytes
from app.config import settings
from app.deps import get_inference_semaphore, require_engine
from app.engines.base import BaseTTSEngine
from app.schemas import SynthesizeRequest

router = APIRouter(prefix="/v1/tts", tags=["tts"])

_ALLOWED_AUDIO_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}


def _validate_text(text: str) -> None:
    if len(text) > settings.max_text_length:
        raise HTTPException(
            status_code=400,
            detail=f"text exceeds max length of {settings.max_text_length} characters",
        )


async def _read_prompt_audio(prompt_audio: UploadFile) -> bytes:
    if prompt_audio.content_type not in _ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported prompt_audio content type: {prompt_audio.content_type}",
        )
    return await prompt_audio.read()


async def _run_inference(engine: BaseTTSEngine, fn, *args, **kwargs) -> bytes:
    semaphore = get_inference_semaphore()
    async with semaphore:
        try:
            audio = await run_in_threadpool(fn, *args, **kwargs)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail="TTS inference failed") from e
    return wav_bytes(audio, engine.sample_rate)


@router.post("")
async def synthesize(req: SynthesizeRequest, engine: BaseTTSEngine = Depends(require_engine)):
    _validate_text(req.text)
    audio = await _run_inference(engine, engine.synthesize, req.text, req.speaker_id, req.speed)
    return Response(content=audio, media_type="audio/wav")


@router.post("/register")
async def register_speaker(
    speaker_id: str = Form(...),
    prompt_text: str = Form(...),
    prompt_audio: UploadFile = File(...),
    engine: BaseTTSEngine = Depends(require_engine),
):
    prompt_bytes = await _read_prompt_audio(prompt_audio)
    semaphore = get_inference_semaphore()
    async with semaphore:
        try:
            await run_in_threadpool(engine.register_speaker, speaker_id, prompt_text, prompt_bytes)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Speaker registration failed") from e
    return {"speaker_id": speaker_id}


@router.post("/clone")
async def clone(
    text: str = Form(...),
    prompt_text: str = Form(...),
    speed: float = Form(1.0),
    prompt_audio: UploadFile = File(...),
    engine: BaseTTSEngine = Depends(require_engine),
):
    _validate_text(text)
    prompt_bytes = await _read_prompt_audio(prompt_audio)
    audio = await _run_inference(engine, engine.clone, text, prompt_bytes, prompt_text, speed)
    return Response(content=audio, media_type="audio/wav")


@router.post("/cross-lingual")
async def cross_lingual(
    text: str = Form(...),
    speed: float = Form(1.0),
    prompt_audio: UploadFile = File(...),
    engine: BaseTTSEngine = Depends(require_engine),
):
    _validate_text(text)
    prompt_bytes = await _read_prompt_audio(prompt_audio)
    audio = await _run_inference(engine, engine.cross_lingual, text, prompt_bytes, speed)
    return Response(content=audio, media_type="audio/wav")


@router.post("/instruct")
async def instruct(
    text: str = Form(...),
    instruct_text: str = Form(...),
    speed: float = Form(1.0),
    prompt_audio: UploadFile = File(...),
    engine: BaseTTSEngine = Depends(require_engine),
):
    _validate_text(text)
    prompt_bytes = await _read_prompt_audio(prompt_audio)
    audio = await _run_inference(engine, engine.instruct, text, instruct_text, prompt_bytes, speed)
    return Response(content=audio, media_type="audio/wav")
