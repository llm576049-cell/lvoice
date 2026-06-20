from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LVOICE_")

    engine: str = "cosyvoice"
    model_dir: str = "/models/CosyVoice2-0.5B"
    model_name: str = "CosyVoice2-0.5B"
    device: str = "auto"  # auto | cuda | cpu
    max_text_length: int = 2000
    max_concurrent_requests: int = 1
    log_level: str = "info"
    # Registered speakers (from /v1/tts/register) are persisted here so they survive
    # a container restart. Point this at a mounted volume in production.
    speaker_store_path: str = "data/spk2info.pt"


settings = Settings()
