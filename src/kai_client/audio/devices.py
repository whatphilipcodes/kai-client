from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Protocol

from kai_client.audio.pcm import duration_ms, samples
from kai_client.utils.config import Audio
from kai_client.utils.logger import Logger

if TYPE_CHECKING:
    import sounddevice


class Capture(Protocol):
    """What the boundary needs from a microphone: frames, and a way to let go of the device."""

    def frames(self) -> AsyncIterator[bytes]: ...

    def close(self) -> None: ...


class Playback(Protocol):
    """What the boundary needs from a speaker: bytes in, and the queue it holds."""

    drops: int

    def push(self, pcm: bytes) -> None: ...

    def set_rate(self, rate: int) -> None: ...

    def queued_ms(self) -> float: ...

    def clear(self) -> None: ...

    def close(self) -> None: ...


def _channels(kind: str) -> str:
    return "max_input_channels" if kind == "input" else "max_output_channels"


def device_names(kind: str) -> list[str]:
    import sounddevice

    return [
        info["name"] for info in sounddevice.query_devices() if info[_channels(kind)]
    ]


def _stop(
    stream: sounddevice.RawInputStream | sounddevice.RawOutputStream | None,
) -> None:
    if stream is None:
        return
    with suppress(Exception):
        stream.stop()
        stream.close()


def _latency(stream: sounddevice.RawInputStream | sounddevice.RawOutputStream) -> float:
    """The device's own round trip, whether it hands back one number or the input/output pair."""
    value = stream.latency
    if isinstance(value, tuple):
        value = value[0]
    return round(float(value) * 1000, 1)


def device_index(name: str, kind: str) -> int | None:
    """None means the system default; a name is a case-insensitive substring of the device name."""
    if not name:
        return None
    import sounddevice

    for index, info in enumerate(sounddevice.query_devices()):
        if info[_channels(kind)] and name.lower() in info["name"].lower():
            return index
    raise ValueError(
        f"no {kind} device matching {name!r}, available: {device_names(kind)}"
    )


class Microphone:
    """A capture device as an async iterator of frames; the newest win if the loop stalls."""

    def __init__(self, cfg: Audio, name: str = "", keep: int = 25) -> None:
        import sounddevice

        self.cfg = cfg
        self.log = Logger()
        self.flags = 0
        self.reported = 0
        self._ready: deque[bytes] = deque(maxlen=keep)
        self._wake: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: sounddevice.RawInputStream | None = None
        stream = sounddevice.RawInputStream(
            samplerate=cfg.input_rate,
            blocksize=samples(cfg.frame_ms, cfg.input_rate),
            channels=1,
            dtype="int16",
            latency="low",
            device=device_index(name or cfg.input_device, "input"),
            callback=self._capture,
        )
        self._stream = stream
        self.log.step(
            "mic.open",
            device=stream.device,
            rate=int(stream.samplerate),
            block=int(stream.blocksize),
            latency_ms=_latency(stream),
        )
        stream.start()

    def _capture(self, indata: Any, _frames: int, _time_info: Any, status: Any) -> None:
        if status:
            self.flags += 1
        self._ready.append(bytes(indata))
        if self._loop is not None:
            with suppress(Exception):
                self._loop.call_soon_threadsafe(self._notify)

    def _notify(self) -> None:
        if self._wake is not None:
            self._wake.set()

    async def frames(self) -> AsyncIterator[bytes]:
        self._wake = asyncio.Event()
        self._loop = asyncio.get_running_loop()
        try:
            while True:
                while self._ready:
                    yield self._ready.popleft()
                if self.flags != self.reported:
                    self.log.warning(
                        "mic.callback", callbacks=self.flags, queued=len(self._ready)
                    )
                    self.reported = self.flags
                self._wake.clear()
                await self._wake.wait()
        finally:
            self.close()

    def close(self) -> None:
        stream, self._stream = self._stream, None
        self.log.detail("mic.close")
        _stop(stream)


class Speaker:
    """A playback device fed by a queue.

    A reply arrives faster than the speaker drains it whenever the TTS beats realtime, so the queue
    holds a whole reply and drops nothing: `max_lag_ms` is a memory guard, not a playout policy. An
    underrun between two chunks of the same reply is silence, and a barge-in clears the queue."""

    def __init__(self, cfg: Audio, rate: int, name: str = "") -> None:
        self.cfg, self.rate, self.name = cfg, rate, name or cfg.output_device
        self.log = Logger()
        self.hold = max(1, int((cfg.prebuffer_ms + cfg.jitter_ms) // cfg.frame_ms))
        self.cap = max(self.hold + 1, int(cfg.max_lag_ms // cfg.frame_ms))
        self._frames: deque[bytes] = deque()
        self._carry = b""
        self._stream: sounddevice.RawOutputStream | None = None
        self._stream = self._open()
        self.drops = 0
        self.played = 0
        self.started = False

    def _open(self) -> sounddevice.RawOutputStream:
        import sounddevice

        stream = sounddevice.RawOutputStream(
            samplerate=self.rate,
            blocksize=samples(self.cfg.frame_ms, self.rate),
            channels=1,
            dtype="int16",
            latency="low",
            device=device_index(self.name, "output"),
            callback=self._play,
        )
        self.log.step(
            "speaker.open",
            device=stream.device,
            rate=int(stream.samplerate),
            block=int(stream.blocksize),
            latency_ms=_latency(stream),
            hold=self.hold,
            cap=self.cap,
        )
        stream.start()
        return stream

    def push(self, pcm: bytes) -> None:
        while len(self._frames) >= self.cap:
            self._frames.popleft()
            self.drops += 1
            if self.drops == 1:
                self.log.warning(
                    "speaker.queue.full",
                    cap=self.cap,
                    queued_ms=round(self.queued_ms(), 1),
                )
        self._frames.append(pcm)

    def _play(self, outdata: Any, _frames: int, _time_info: Any, _status: Any) -> None:
        data = bytearray(len(outdata))
        if not self.started and len(self._frames) >= self.hold:
            self.started = True
        pos = 0
        if self.started:
            while pos < len(data) and (self._carry or self._frames):
                chunk = self._carry or self._frames.popleft()
                take = min(len(data) - pos, len(chunk))
                data[pos : pos + take] = chunk[:take]
                self._carry = chunk[take:]
                pos += take
                self.played += take
        outdata[: len(data)] = bytes(data)

    def set_rate(self, rate: int) -> None:
        if rate == self.rate:
            return
        self.log.step("speaker.rate", from_rate=self.rate, to_rate=rate)
        self.rate = rate
        _stop(self._stream)
        self._stream = self._open()
        self.clear()

    def queued_ms(self) -> float:
        held = sum(len(frame) for frame in self._frames) + len(self._carry)
        return duration_ms(held, self.rate)

    def clear(self) -> None:
        if self._frames or self._carry:
            self.log.detail(
                "speaker.clear",
                frames=len(self._frames),
                queued_ms=round(self.queued_ms(), 1),
            )
        self._frames.clear()
        self._carry = b""
        self.started = False

    def close(self) -> None:
        stream, self._stream = self._stream, None
        self.log.step(
            "speaker.close",
            played_s=round(duration_ms(self.played, self.rate) / 1000, 1),
            drops=self.drops,
        )
        _stop(stream)
