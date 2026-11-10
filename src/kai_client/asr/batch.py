from __future__ import annotations

import httpx

from kai_client.audio.pcm import duration_ms, wav_bytes
from kai_client.utils.config import Asr
from kai_client.utils.logger import Logger


class Batch:
    def __init__(self, config: Asr, rate: int) -> None:
        self.config = config
        self.rate = rate
        self.client = httpx.AsyncClient(
            base_url=config.url,
            timeout=httpx.Timeout(30.0, connect=2.0),
            limits=httpx.Limits(max_connections=4, keepalive_expiry=60.0),
            headers={"authorization": f"Bearer {config.api_key}"}
            if config.api_key
            else {},
        )

    def session(self) -> BatchAsr:
        return BatchAsr(self)

    async def aclose(self) -> None:
        await self.client.aclose()


class BatchAsr:
    """The same question asked again: a batch server keeps no session, so every read posts the
    speech so far and the turn's last read is its transcript."""

    def __init__(self, stt: Batch) -> None:
        self.asr = stt
        self.config = stt.config
        self.log = Logger()
        self.pcm = b""
        self.unreachable = False

    def push(self, pcm: bytes) -> None:
        self.pcm = pcm

    async def text(self) -> str:
        data = {"model": self.config.model}
        if self.config.language:
            data["language"] = self.config.language
        try:
            response = await self.asr.client.post(
                "/audio/transcriptions",
                data=data,
                files={
                    "file": (
                        "turn.wav",
                        wav_bytes(self.pcm, self.asr.rate),
                        "audio/wav",
                    )
                },
            )
            response.raise_for_status()
        except httpx.TransportError as exc:
            self.unreachable = True
            raise ConnectionError(f"{type(exc).__name__}: {exc}") from exc
        text = response.json().get("text", "")
        self.log.detail(
            "asr.posted",
            http=response.status_code,
            audio_ms=round(duration_ms(len(self.pcm), self.asr.rate), 1),
            text=text,
        )
        return text

    async def finish(self) -> str:
        return await self.text()

    def abort(self) -> None:
        pass
