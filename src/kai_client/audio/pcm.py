from __future__ import annotations

import array
import math
import struct

WIDTH = 2


def samples(ms: float, rate: int) -> int:
    return round(ms * 0.001 * rate)


def duration_ms(count: int, rate: int) -> float:
    return count / WIDTH / rate * 1000


def frame_bytes(ms: float, rate: int) -> int:
    return samples(ms, rate) * WIDTH


def rms16(pcm: bytes) -> float:
    raw = array.array("h")
    raw.frombytes(pcm[: len(pcm) - len(pcm) % WIDTH])
    if not raw:
        return 0.0
    return math.sqrt(sum(s * s for s in raw) / len(raw)) / 32768.0


def wav_bytes(pcm: bytes, rate: int, channels: int = 1) -> bytes:
    data = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        rate,
        rate * channels * WIDTH,
        channels * WIDTH,
        16,
        b"data",
        data,
    )
    return header + pcm
