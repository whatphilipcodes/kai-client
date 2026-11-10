from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic_ns
from typing import Any

from kai_client.runtime.bus import Events
from kai_client.runtime.protocol import Kind, encode_info

POINTS = (
    "turn_start",
    "mic_end",
    "turn_open",
    "stt_start",
    "stt_end",
    "llm_start",
    "llm_first",
    "llm_end",
    "tts_start",
    "tts_first",
    "play_first",
    "play_end",
)

DERIVED = (
    ("vad_ms", "turn_open", "mic_end"),
    ("stt_ms", "stt_end", "stt_start"),
    ("llm_ttft_ms", "llm_first", "llm_start"),
    ("llm_ms", "llm_end", "llm_start"),
    ("tts_ttfa_ms", "tts_first", "tts_start"),
    ("stream_ms", "play_end", "tts_first"),
    ("e2e_ms", "play_first", "mic_end"),
    ("turn_ms", "play_end", "mic_end"),
    ("prelook_ms", "mic_end", "llm_start"),
)


@dataclass(slots=True)
class TurnTrace:
    turn: int
    epoch: int
    points: dict[str, int] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    done: bool = False

    def add(self, point: str, ns: int) -> None:
        self.points.setdefault(point, ns)

    def derived(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, end, start in DERIVED:
            if end in self.points and start in self.points:
                out[name] = (self.points[end] - self.points[start]) / 1e6
        return out

    def line(self) -> str:
        parts = [f"turn={self.turn}" + (f" ep={self.epoch}" if self.epoch else "")]
        if "status" in self.labels:
            parts.append(f"status={self.labels['status']}")
        for name, value in self.derived().items():
            parts.append(f"{name}={value:.1f}")
        parts.append(
            f"stages={sum(point in POINTS for point in self.points)}/{len(POINTS)}"
        )
        return " ".join(parts)


def metrics(line: str) -> dict[str, float]:
    """The numbers out of a turn line, which is how a run is read back without the trace objects."""
    out: dict[str, float] = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, _, value = token.partition("=")
        try:
            out[key] = float(value)
        except ValueError:
            continue
    return out


class Tracer:
    def __init__(
        self,
        events: Events,
        enabled: bool = True,
        sink: Callable[[str, int, int, int, dict[str, Any]], None] | None = None,
    ) -> None:
        self.events = events
        self.enabled = enabled
        self.sink = sink

    def mark(
        self, point: str, turn: int, epoch: int, ns: int | None = None, **labels: Any
    ) -> None:
        if not self.enabled:
            return
        stamp = monotonic_ns() if ns is None else ns
        if self.sink is not None:
            self.sink(point, turn, epoch, stamp, labels)
            return
        self.events.send(
            encode_info(
                Kind.TRACE, turn, epoch, {"point": point, "ns": stamp, **labels}
            )
        )


class TraceCollector:
    KEEP = 64

    def __init__(self) -> None:
        self.turns: dict[tuple[int, int], TurnTrace] = {}

    def record(
        self,
        point: str,
        turn: int,
        epoch: int,
        ns: int,
        labels: dict[str, Any] | None = None,
    ) -> TurnTrace | None:
        key = (turn, epoch)
        trace = self.turns.get(key)
        if trace is None:
            if point == "turn_end":
                return None
            trace = self.turns.setdefault(key, TurnTrace(turn, epoch))
        elif trace.done:
            return None
        if point == "turn_end":
            trace.done = True
            trace.labels.update(labels or {})
        else:
            trace.add(point, ns)
        if len(self.turns) > self.KEEP:
            oldest = next((key for key, held in self.turns.items() if held.done), None)
            if oldest is not None:
                del self.turns[oldest]
        return trace if trace.done else None
