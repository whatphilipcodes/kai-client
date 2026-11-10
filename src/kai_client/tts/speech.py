from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from pathlib import Path
from time import monotonic_ns

import httpx

from kai_client.audio.pcm import duration_ms
from kai_client.utils.config import Tts, asset
from kai_client.utils.logger import Logger

_NO_REGISTRY = (404, 405, 501)


class Speech:
    def __init__(self, config: Tts) -> None:
        self.config = config
        self.log = Logger()
        self._step = max(160, int(config.chunk_ms * 0.001 * config.sample_rate) * 2)
        self._rate = config.sample_rate
        ref = asset(config.ref_audio) if config.ref_audio else None
        self._ref_path = ref
        self._ref_audio = _data_url(ref) if ref is not None else ""
        self._ref_text = _text(asset(config.ref_text)) if config.ref_text else ""
        self._voice = config.voice
        self._registry: bool | None = None
        self._tried = False
        self.client = httpx.AsyncClient(
            base_url=config.url,
            timeout=httpx.Timeout(120.0, connect=2.0),
            limits=httpx.Limits(max_connections=4, keepalive_expiry=120.0),
            headers={"authorization": f"Bearer {config.api_key}"}
            if config.api_key
            else {},
        )

    @property
    def sample_rate(self) -> int:
        return self._rate

    async def register(self) -> None:
        """Hand the reference clip to the server's voice registry, once.

        A server that is still loading gets asked again at the next synthesis; a
        server with no registry is not asked a second time, and every request keeps
        carrying the clip inline.
        """
        if self._ref_path is None or self._registry is not None:
            return
        name = (
            self._ref_path.stem
            if self.config.voice in ("", "default")
            else self.config.voice
        )
        try:
            listed = await self.client.get("/audio/voices")
            listed.raise_for_status()
            registry = listed.json()
            known = {str(v).lower() for v in registry.get("voices", [])} | {
                str(item.get("name", "")).lower()
                for item in registry.get("uploaded_voices", [])
            }
            if name.lower() not in known:
                response = await self.client.post(
                    "/audio/voices",
                    data={"consent": "local", "name": name, "ref_text": self._ref_text},
                    files={
                        "audio_sample": (
                            self._ref_path.name,
                            self._ref_path.read_bytes(),
                            "audio/wav",
                        )
                    },
                )
                response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            missing = (
                isinstance(exc, httpx.HTTPStatusError)
                and exc.response.status_code in _NO_REGISTRY
            )
            self._registry = False if missing else None
            if self._tried:
                self.log.detail(
                    "tts.voice",
                    name=name,
                    error=f"{type(exc).__name__}: {exc}",
                    inline=True,
                )
            else:
                self.log.warning(
                    "tts.voice",
                    name=name,
                    error=f"{type(exc).__name__}: {exc}",
                    inline=True,
                )
            self._tried = True
            return
        self._registry, self._voice, self._ref_audio, self._ref_text = (
            True,
            name,
            "",
            "",
        )
        self.log.step("tts.voice", name=name, bytes=self._ref_path.stat().st_size)

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        await self.register()
        body = {
            "input": text,
            "model": self.config.model,
            "voice": self._voice,
            "response_format": "pcm",
            "stream": True,
        }
        if self.config.stream_format:
            body["stream_format"] = self.config.stream_format
        if self._ref_audio:
            body["ref_audio"] = self._ref_audio
        if self._ref_text:
            body["ref_text"] = self._ref_text
        began = monotonic_ns()
        first, chunks, bytes_out, wire_bytes = 0.0, 0, 0, 0
        self.log.step(
            "tts.request",
            model=self.config.model,
            chars=len(text),
            chunk_bytes=self._step,
            ref=bool(self._ref_audio),
        )
        try:
            async with self.client.stream(
                "POST", "/audio/speech", json=body
            ) as response:
                response.raise_for_status()
                self._adopt(response.headers.get("x-audio-sample-rate", ""))
                buffered = bytearray()
                async for block in _blocks(response):
                    wire_bytes += len(block)
                    buffered += block
                    while len(buffered) >= self._step:
                        chunks += 1
                        first = first or round((monotonic_ns() - began) / 1e6, 1)
                        bytes_out += self._step
                        self.log.detail(
                            "tts.chunk",
                            n=chunks,
                            ms=round((monotonic_ns() - began) / 1e6, 1),
                        )
                        yield bytes(buffered[: self._step])
                        del buffered[: self._step]
                if buffered:
                    chunks += 1
                    first = first or round((monotonic_ns() - began) / 1e6, 1)
                    bytes_out += len(buffered)
                    yield bytes(buffered)
        finally:
            self.log.step(
                "tts.audio",
                first_ms=first,
                ms=round((monotonic_ns() - began) / 1e6, 1),
                chunks=chunks,
                bytes=bytes_out,
                audio_s=round(duration_ms(bytes_out, self._rate) / 1000, 2),
                wire_bytes=wire_bytes,
                rate=self._rate,
            )

    def _adopt(self, rate: str) -> None:
        if rate.isdigit() and int(rate) != self._rate:
            self._rate = int(rate)
            self._step = max(160, int(self.config.chunk_ms * 0.001 * self._rate) * 2)

    async def aclose(self) -> None:
        await self.client.aclose()


async def _blocks(response: httpx.Response) -> AsyncIterator[bytes]:
    if "text/event-stream" not in response.headers.get("content-type", ""):
        async for block in response.aiter_bytes(1 << 15):
            yield block
        return
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        text = line[5:].strip()
        if not text.startswith("{"):
            continue
        event = json.loads(text)
        match event.get("type"):
            case "speech.audio.delta":
                yield base64.b64decode(event.get("audio", ""))
            case "speech.audio.error":
                raise RuntimeError(event["error"]["message"])


def _data_url(path: Path) -> str:
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode()


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()
