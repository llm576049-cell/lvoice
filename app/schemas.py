from pydantic import BaseModel, Field


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1)
    speaker_id: str
    speed: float = Field(default=1.0, gt=0, le=3.0)


class InstructRequest(BaseModel):
    text: str = Field(min_length=1)
    instruct_text: str = Field(min_length=1)
    speed: float = Field(default=1.0, gt=0, le=3.0)


class RegisterSpeakerRequest(BaseModel):
    speaker_id: str = Field(min_length=1)
    prompt_text: str = Field(min_length=1)


class VoicesResponse(BaseModel):
    voices: list[str]


class ErrorResponse(BaseModel):
    detail: str
