from __future__ import annotations

import contextlib
import json
from base64 import b64encode
from collections.abc import AsyncIterator
from time import monotonic_ns

import httpx

from kai_client.audio.pcm import wav_bytes
from kai_client.utils.config import Llm
from kai_client.utils.logger import Logger


class Chat:
    def __init__(self, config: Llm) -> None:
        self.config = config
        self.log = Logger()
        self.client = httpx.AsyncClient(
            base_url=config.url,
            timeout=httpx.Timeout(120.0, connect=2.0),
            limits=httpx.Limits(max_connections=8, keepalive_expiry=120.0),
            headers={"authorization": f"Bearer {config.api_key}"}
            if config.api_key
            else {},
        )

    async def stream(
        self, messages: list[dict], *, max_tokens: int | None = None
    ) -> AsyncIterator[str]:
        limit = self.config.max_tokens if max_tokens is None else max_tokens
        body = {
            "model": self.config.model,
            "messages": _wire(messages),
            "stream": True,
            "temperature": self.config.temperature,
            "max_tokens": limit,
        }
        began = monotonic_ns()
        first, deltas, chars = 0.0, 0, 0
        now = lambda: round((monotonic_ns() - began) / 1e6, 1)
        ask = self.log.detail if limit <= 1 else self.log.step
        ask(
            "llm.request",
            model=self.config.model,
            messages=len(messages),
            audio=sum(1 for msg in messages if isinstance(msg.get("content"), list)),
            prefill=limit <= 1 or None,
            said=str(messages[-1].get("content"))[:160],
        )
        try:
            async with self.client.stream(
                "POST", "/chat/completions", json=body
            ) as response:
                response.raise_for_status()
                self.log.detail("llm.answered", http=response.status_code, ms=now())
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if not chunk or chunk == "[DONE]":
                        if chunk == "[DONE]":
                            return
                        continue
                    delta = json.loads(chunk)["choices"][0].get("delta") or {}
                    if delta.get("content"):
                        deltas += 1
                        chars += len(delta["content"])
                        first = first or round((monotonic_ns() - began) / 1e6, 1)
                        self.log.detail("llm.delta", ms=now(), text=delta["content"])
                        yield delta["content"]
        finally:
            ask("llm.answer", first_ms=first, ms=now(), deltas=deltas, chars=chars)

    async def prewarm(self) -> None:
        with contextlib.suppress(httpx.HTTPError):
            await self.client.get("/models")

    async def aclose(self) -> None:
        await self.client.aclose()


def _wire(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        out.append(
            {**msg, "content": [_part(p) for p in content]}
            if isinstance(content, list)
            else msg
        )
    return out


def _part(part: dict) -> dict:
    """Audio goes out as `audio_url`, the shape the model's serving recipe documents.

    `input_audio` is the other OpenAI spelling and a template that only looks for `audio_url`
    will render the turn without ever placing the audio in it.
    """
    if part.get("type") != "audio":
        return part
    wav = b64encode(wav_bytes(part["pcm"], part["rate"])).decode()
    return {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{wav}"}}
