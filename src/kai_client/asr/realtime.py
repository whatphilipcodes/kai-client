from __future__ import annotations

import asyncio
import base64
import json
from contextlib import suppress
from time import monotonic_ns
from typing import Any

import websockets

from kai_client.audio.pcm import duration_ms
from kai_client.utils.config import Asr
from kai_client.utils.logger import Logger

DONE_TIMEOUT_S = 15.0
COMMIT_EVENT = {"type": "input_audio_buffer.commit", "final": True}
START_EVENT = {"type": "input_audio_buffer.commit", "final": False}
CREATED = "session.created"
NOTHING = "the ASR service took the turn's commit and answered it with nothing"


class Realtime:
    def __init__(self, config: Asr, rate: int) -> None:
        self.config = config
        self.rate = rate
        self.log = Logger()

    def session(self) -> RealtimeSession:
        return RealtimeSession(self.config, self.rate)

    async def aclose(self) -> None:
        pass


class RealtimeSession:
    """One turn's speech, streamed to the ASR service while it is still being said.

    `push` takes the audio of the turn so far and sends only what has not been sent, so a tick that
    fires twice costs nothing and both backends are driven the same way. The session lives for the
    turn and no longer: vLLM does not preempt an idle streaming session, so a socket left open keeps
    its slot out of the pool for the whole conversation.

    The wait for a transcript starts once the commit has reached the socket, not once it was queued:
    the audio of a turn whose endpoint arrived mid-utterance is still on its way, and a deadline
    measured against the queue would give the service less time the longer the turn was.
    """

    def __init__(self, config: Asr, rate: int) -> None:
        self.config = config
        self.rate = rate
        self.log = Logger()
        self.sent = 0
        self.began = 0.0
        self.heard = ""
        self.error = ""
        self.unreachable = False
        self.out: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.finished = asyncio.Event()
        self.created = asyncio.Event()
        self._task: asyncio.Task | None = None

    def push(self, pcm: bytes) -> None:
        if len(pcm) > self.sent:
            self._start()
            self.out.put_nowait(_append(pcm[self.sent :]))
            self.log.detail(
                "asr.queued", bytes=len(pcm) - self.sent, total_ms=self.audio_ms
            )
            self.sent = len(pcm)

    async def text(self) -> str:
        if self.error:
            raise RuntimeError(self.error)
        return self.heard

    async def finish(self) -> str:
        self._start()
        self.out.put_nowait(COMMIT_EVENT)
        until = asyncio.get_running_loop().time() + DONE_TIMEOUT_S
        self.log.step(
            "asr.commit", sent_ms=round(self._ms(), 1), audio_ms=self.audio_ms
        )
        await self._on_the_wire(max(0.0, until - asyncio.get_running_loop().time()))
        with suppress(TimeoutError):
            await asyncio.wait_for(
                self.finished.wait(),
                max(0.0, until - asyncio.get_running_loop().time()),
            )
        self.abort()
        if self.heard.strip():
            return self.heard
        if self.error:
            if not self.unreachable:
                self.log.error(
                    "asr.error", error=self.error, waited_ms=round(self._ms(), 1)
                )
            raise RuntimeError(self.error)
        self.log.error(
            "asr.silent", audio_ms=self.audio_ms, waited_ms=round(self._ms(), 1)
        )
        raise RuntimeError(NOTHING)

    async def _on_the_wire(self, timeout: float) -> None:
        """The service cannot answer a commit it has not received, and a session that has already
        ended is never going to send it, so the drain waits for whichever of the two comes first."""
        pending = [
            asyncio.ensure_future(self.out.join()),
            asyncio.ensure_future(self.finished.wait()),
        ]
        await asyncio.wait(
            pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        for waiting in pending:
            waiting.cancel()

    def abort(self) -> None:
        if self._task is not None:
            self.log.detail("asr.socket.close", sent_ms=round(self._ms(), 1))
            self._task.cancel()
            self._task = None

    def _start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="asr.session")

    async def _run(self) -> None:
        headers = (
            {"authorization": f"Bearer {self.config.api_key}"}
            if self.config.api_key
            else {}
        )
        self.log.step("asr.connect", url=self.config.url, model=self.config.model)
        try:
            async with websockets.connect(
                self.config.url,
                additional_headers=headers,
                max_size=None,
                open_timeout=2.0,
            ) as socket:
                self.began = monotonic_ns()
                reader = asyncio.create_task(self._read(socket), name="asr.read")
                try:
                    await self._validate(socket)
                    while True:
                        body = await self.out.get()
                        await socket.send(json.dumps(body))
                        self.log.detail("asr.sent", kind=body["type"])
                        self.out.task_done()
                finally:
                    reader.cancel()
                    with suppress(asyncio.CancelledError):
                        await reader
        except Exception as exc:  # noqa: BLE001
            self.unreachable = isinstance(exc, OSError)
            self.log.error(
                "asr.unreachable",
                error=f"{type(exc).__name__}: {exc}",
                url=self.config.url,
            )
            self.error = f"{type(exc).__name__}: {exc}"
            self.finished.set()

    def _ms(self) -> float:
        return 0.0 if not self.began else (monotonic_ns() - self.began) / 1e6

    @property
    def audio_ms(self) -> float:
        return round(duration_ms(self.sent, self.rate), 1)

    async def _validate(self, socket: websockets.ClientConnection) -> None:
        """The service speaks first: vLLM only binds the model once its own session is open."""
        with suppress(TimeoutError):
            await asyncio.wait_for(self.created.wait(), 2.0)
        if not self.created.is_set():
            self.log.warning("asr.no_session_created", waited_ms=2000)
        await socket.send(
            json.dumps({"type": "session.update", "model": self.config.model})
        )
        await socket.send(json.dumps(START_EVENT))
        self.log.detail("asr.start_commit")

    async def _read(self, socket: websockets.ClientConnection) -> None:
        try:
            async for raw in socket:
                self._absorb(json.loads(raw))
        except websockets.ConnectionClosed:
            self.log.warning(
                "asr.socket.closed", heard=self.heard, waited_ms=round(self._ms(), 1)
            )
            self.finished.set()

    def _absorb(self, event: dict[str, Any]) -> None:
        kind = str(event.get("type", ""))
        self.log.detail("asr.event", type=kind, ms=round(self._ms(), 1))
        if kind == CREATED:
            self.created.set()
        elif kind.endswith("transcription.delta"):
            self.heard = _merged(self.heard, event)
            self.log.detail("asr.heard", ms=round(self._ms(), 1), text=self.heard)
        elif kind.endswith(("transcription.done", "transcription.completed")):
            self.heard = str(event.get("text") or event.get("transcript") or self.heard)
            usage = event.get("usage") or {}
            self.log.step(
                "asr.done",
                ms=round(self._ms(), 1),
                text=self.heard,
                prompt=usage.get("prompt_tokens"),
                tokens=usage.get("completion_tokens"),
            )
            self.finished.set()
        elif kind == "error":
            self.error = str(
                event.get("message")
                or event.get("error")
                or event.get("code")
                or "error"
            )
            self.log.error("asr.event.error", code=event.get("code"), error=self.error)
            self.finished.set()


def _append(pcm: bytes) -> dict[str, Any]:
    return {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(pcm).decode("ascii"),
    }


def _merged(heard: str, event: dict[str, Any]) -> str:
    """vLLM's delta events carry the new words; OpenAI's carry the transcript so far."""
    if event.get("delta") is not None:
        return heard + str(event["delta"])
    return str(event.get("text", "")) + str(event.get("stash") or "")
