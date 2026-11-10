from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from kai_client.audio.pcm import duration_ms, rms16
from kai_client.utils.config import Vad
from kai_client.utils.logger import Logger


class Gate(Protocol):
    speech: bool
    turn: int

    def feed(self, pcm: bytes, ns: int) -> list[Event]: ...

    def flush(self) -> SpeechEnd | None: ...

    def voiced_pcm(self) -> bytes: ...

    def voiced_ms(self) -> float: ...


@dataclass(slots=True)
class SpeechStart:
    turn: int
    ns: int


@dataclass(slots=True)
class SpeechPending:
    turn: int
    ns: int


@dataclass(slots=True)
class SpeechEnd:
    turn: int
    ns: int
    voiced_end_ns: int
    pcm: bytes
    duration_ms: float
    trailing_silence_ms: float
    voiced_len: int = 0


Event = SpeechStart | SpeechPending | SpeechEnd


@dataclass
class EnergyVad:
    config: Vad
    sample_rate: int
    frame_bytes: int
    speech: bool = False
    turn: int = 0
    _ring: list[tuple[bytes, int]] = field(default_factory=list)
    _voiced: bytearray = field(default_factory=bytearray)
    _voiced_end_off: int = 0
    _voiced_end_ns: int = 0
    _speech_run: int = 0
    _silence_run: int = 0
    _pending_off: int = -1

    def __post_init__(self) -> None:
        rate = self.sample_rate
        self.pad_bytes = int(self.config.speech_pad_ms * 0.001 * rate) * 2
        self.pre_frames = max(1, self.pad_bytes // max(1, self.frame_bytes) + 1)
        self.min_speech_bytes = max(
            self.frame_bytes, int(self.config.min_speech_ms * 0.001 * rate) * 2
        )
        self.min_silence_bytes = max(
            self.frame_bytes, int(self.config.min_silence_ms * 0.001 * rate) * 2
        )

    def score(self, pcm: bytes) -> float:
        return rms16(pcm)

    def voiced_ms(self) -> float:
        return duration_ms(self._voiced_end_off, self.sample_rate)

    def voiced_pcm(self) -> bytes:
        return bytes(self._voiced[: self._voiced_end_off])

    def feed(self, pcm: bytes, ns: int) -> list[Event]:
        self._ring.append((pcm, ns))
        if len(self._ring) > self.pre_frames:
            self._ring.pop(0)
        score = self.score(pcm)
        if not self.speech:
            return self._idle(score, pcm, ns)
        return self._active(score, pcm, ns)

    def _idle(self, score: float, pcm: bytes, ns: int) -> list[Event]:
        if score < self.config.rms_open:
            if score < self.config.rms_close:
                self._speech_run = 0
            return []
        self._speech_run += len(pcm)
        if self._speech_run < self.min_speech_bytes:
            return []
        self.turn += 1
        self.speech = True
        self._pending_off = -1
        self._voiced = bytearray(b"".join(frame for frame, _ in self._ring))
        self._voiced_end_off = len(self._voiced)
        self._voiced_end_ns = ns
        self._speech_run = 0
        self._silence_run = 0
        return [SpeechStart(self.turn, ns)]

    def _active(self, score: float, pcm: bytes, ns: int) -> list[Event]:
        if score >= self.config.rms_close:
            self._voiced += pcm
            self._voiced_end_off = len(self._voiced)
            self._voiced_end_ns = ns
            self._silence_run = 0
            return []
        pending: list[Event] = []
        if self._silence_run == 0 and self._voiced_end_off > self._pending_off:
            self._pending_off = self._voiced_end_off
            pending = [self._pending(ns)]
        self._voiced += pcm
        self._silence_run += len(pcm)
        if self._silence_run < self.min_silence_bytes:
            return pending
        return [*pending, self._close(ns)]

    def _pending(self, ns: int) -> SpeechPending:
        return SpeechPending(self.turn, ns)

    def _close(self, ns: int, tail_bytes: int | None = None) -> SpeechEnd:
        tail = (
            min(self._silence_run, self.pad_bytes) if tail_bytes is None else tail_bytes
        )
        segment = bytes(self._voiced[: self._voiced_end_off + tail])
        end = SpeechEnd(
            turn=self.turn,
            ns=ns,
            voiced_end_ns=self._voiced_end_ns,
            pcm=segment,
            duration_ms=self.voiced_ms(),
            trailing_silence_ms=duration_ms(tail, self.sample_rate),
            voiced_len=self._voiced_end_off,
        )
        self.speech = False
        self._voiced = bytearray()
        self._silence_run = 0
        return end

    def flush(self) -> SpeechEnd | None:
        if not self.speech or not self._voiced:
            return None
        return self._close(self._voiced_end_ns, tail_bytes=0)


@dataclass
class ManualVad:
    config: Vad
    sample_rate: int
    frame_bytes: int
    speech: bool = False
    turn: int = 0
    _voiced: bytearray = field(default_factory=bytearray)
    _last_ns: int = 0
    log: Logger = field(default_factory=Logger)

    def __post_init__(self) -> None:
        self.min_speech_bytes = max(
            self.frame_bytes,
            int(self.config.min_speech_ms * 0.001 * self.sample_rate) * 2,
        )

    def voiced_ms(self) -> float:
        return duration_ms(len(self._voiced), self.sample_rate)

    def voiced_pcm(self) -> bytes:
        return bytes(self._voiced)

    def feed(self, pcm: bytes, ns: int) -> list[Event]:
        self._last_ns = ns
        if self.speech:
            self._voiced += pcm
        return []

    def toggle(self) -> list[Event]:
        if self.speech:
            end = self._end()
            return [end] if end is not None else []
        self.turn += 1
        self.speech = True
        self._voiced = bytearray()
        return [SpeechStart(self.turn, self._last_ns)]

    def flush(self) -> SpeechEnd | None:
        return self._end() if self.speech else None

    def _end(self) -> SpeechEnd | None:
        voiced_ms = self.voiced_ms()
        if len(self._voiced) < self.min_speech_bytes:
            self.log.warning(
                "gate.short",
                turn=self.turn,
                voiced_ms=round(voiced_ms, 1),
                min_speech_ms=self.config.min_speech_ms,
            )
            self.speech = False
            self._voiced = bytearray()
            return None
        end = SpeechEnd(
            turn=self.turn,
            ns=self._last_ns,
            voiced_end_ns=self._last_ns,
            pcm=bytes(self._voiced),
            duration_ms=voiced_ms,
            trailing_silence_ms=0.0,
            voiced_len=len(self._voiced),
        )
        self.speech = False
        self._voiced = bytearray()
        return end


def build(config: Vad, sample_rate: int, frame_bytes: int) -> Gate:
    if config.engine == "energy":
        return EnergyVad(
            config=config, sample_rate=sample_rate, frame_bytes=frame_bytes
        )
    if config.engine == "manual":
        return ManualVad(
            config=config, sample_rate=sample_rate, frame_bytes=frame_bytes
        )
    raise ValueError(
        f"vad.engine={config.engine!r} needs a model; this build ships the RMS gate and the space bar only. "
        "Point EnergyVad.score at a frame-level VAD to add one."
    )
