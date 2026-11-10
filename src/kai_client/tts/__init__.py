from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from kai_client.utils.config import Config


class TtsStream(Protocol):
    """The speech service: text in, pcm out, at a rate the synthesizer has to follow."""

    @property
    def sample_rate(self) -> int: ...

    def stream(self, text: str) -> AsyncIterator[bytes]: ...

    async def register(self) -> None: ...

    async def aclose(self) -> None: ...


def build(cfg: Config) -> TtsStream:
    from kai_client.tts.speech import Speech

    return Speech(cfg.tts)
