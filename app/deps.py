from app.engines.base import BaseTTSEngine

_engine: BaseTTSEngine | None = None


def set_engine(engine: BaseTTSEngine) -> None:
    global _engine
    _engine = engine


def get_engine() -> BaseTTSEngine:
    if _engine is None:
        raise RuntimeError("Engine not initialized")
    return _engine
