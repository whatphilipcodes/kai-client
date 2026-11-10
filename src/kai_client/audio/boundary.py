from __future__ import annotations

import asyncio
from collections.abc import Callable
from time import monotonic_ns
from typing import Any

from kai_client.audio import devices, pcm
from kai_client.runtime import wire
from kai_client.runtime.protocol import FLAG_FINAL, Kind, Msg
from kai_client.runtime.stage import Stage
from kai_client.runtime.trace import TraceCollector, Tracer
from kai_client.terminal import console
from kai_client.utils.config import Config

PLAYING_MAX = 8


class Boundary(Stage):
    name = "audio"
    event_topics = (
        Kind.HELLO,
        Kind.STATE,
        Kind.CANCEL,
        Kind.TRACE,
        Kind.TOKEN,
        Kind.TURN,
    )

    def __init__(
        self,
        cfg: Config,
        *,
        mic: devices.Capture | None = None,
        speaker: devices.Playback | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(cfg)
        self.mic = mic if mic is not None else devices.Microphone(cfg.audio)
        self.speaker = (
            speaker
            if speaker is not None
            else devices.Speaker(cfg.audio, cfg.tts.sample_rate)
        )
        self.on_line = on_line
        self.audio_in = wire.UdpReceiver(cfg.network.udp("tts"))
        self.audio_out = wire.UdpSender(cfg.network.udp("mic"))
        self.seq = 0
        self.speaking = False
        self.playing: dict[tuple[int, int], int] = {}
        self.trace_log = TraceCollector()
        self.tracer = Tracer(self.events_out, cfg.trace.enabled, sink=self._record)
        self.io = console.Dialogue()

    async def start(self) -> None:
        await super().start()
        self.audio_in.attach(asyncio.get_running_loop(), self._on_audio)
        self.log.step(
            "devices",
            mic=type(self.mic).__name__,
            speaker=type(self.speaker).__name__,
            rate=self.cfg.audio.input_rate,
        )
        self.log.step(
            "udp", in_port=self.audio_in.port, mic_port=self.cfg.network.udp("mic").port
        )
        self.tasks.append(asyncio.create_task(self._pump_mic(), name="audio.mic"))

    async def stop(self) -> None:
        await super().stop()
        self.mic.close()
        self.speaker.close()
        self.audio_in.close()
        self.audio_out.close()

    def state(self) -> dict[str, Any]:
        queued = self.speaker.queued_ms() > 0
        return {"speaking": self.speaking or queued, "turn": self.guard.turn}

    async def _pump_mic(self) -> None:
        rate = self.cfg.audio.input_rate
        size = pcm.frame_bytes(self.cfg.audio.frame_ms, rate)
        async for frame in self.mic.frames():
            self.seq += 1
            if len(frame) > size:
                self.log.warning(
                    "mic.frame.truncated", seq=self.seq, bytes=len(frame), sent=size
                )
            self.log.detail("mic.frame", seq=self.seq, bytes=min(len(frame), size))
            self.audio_out.send(
                wire.encode(wire.MIC, 0, 0, self.seq, rate, frame[:size])
            )

    def _on_audio(self, packet: wire.Packet) -> None:
        if packet.stream != wire.TTS:
            return
        if not self.guard.adopt(packet.turn, packet.epoch):
            self.log.detail(
                "audio.stale",
                packet.turn,
                packet.epoch,
                seq=packet.seq,
                now=self.guard.turn,
            )
            return
        key = (packet.turn, packet.epoch)
        self.log.detail(
            "audio.receive",
            packet.turn,
            packet.epoch,
            seq=packet.seq,
            bytes=len(packet.pcm),
            flags=packet.flags,
        )
        if packet.flags & wire.LAST:
            self._end_turn(*key, status="spoken")
            return
        if not len(packet.pcm):
            return
        if key in self.playing:
            self.speaker.push(bytes(packet.pcm))
        else:
            self._start_turn(packet)
        self.speaking = True

    def _start_turn(self, packet: wire.Packet) -> None:
        self.playing[packet.turn, packet.epoch] = packet.sent_ns
        if len(self.playing) > PLAYING_MAX:
            dropped = min(self.playing)
            self.log.warning(
                "turn.forgotten",
                turn=dropped[0],
                epoch=dropped[1],
                reason="that many turns were playing at once",
            )
            del self.playing[dropped]
        self.speaker.set_rate(packet.rate)
        self.speaker.push(bytes(packet.pcm))
        queued = round(self.speaker.queued_ms(), 1)
        self.log.step(
            "play.start", packet.turn, packet.epoch, rate=packet.rate, queued_ms=queued
        )
        self.tracer.mark(
            "play_first", packet.turn, packet.epoch, ns=self._play_ns(packet)
        )

    def _end_turn(self, turn: int, epoch: int, status: str) -> None:
        self.playing.pop((turn, epoch), None)
        self.log.step("play.end", turn, epoch, status=status, drops=self.speaker.drops)
        self.tracer.mark(
            "play_end",
            turn,
            epoch,
            ns=monotonic_ns() + int(self.speaker.queued_ms() * 1e6),
        )
        self.tracer.mark("turn_end", turn, epoch, status=status)
        self.speaking = False

    def _play_ns(self, packet: wire.Packet) -> int:
        budget = self.cfg.audio.jitter_ms + self.cfg.audio.prebuffer_ms
        return packet.sent_ns + int(budget * 1e6)

    def on_cancel(self, msg: Msg) -> None:
        if not self.guard.adopt(msg.turn, msg.epoch):
            self.log.detail("cancel.stale", msg.turn, msg.epoch, now=self.guard.turn)
            return
        self.log.step(
            "speaker.clear",
            msg.turn,
            msg.epoch,
            queued_ms=round(self.speaker.queued_ms(), 1),
        )
        self.speaker.clear()
        self._cut_before(msg.turn, msg.epoch)

    def _cut_before(self, turn: int, epoch: int) -> None:
        """Every older turn the boundary was told about but never played out ends here.

        The guard alone cannot name them: a reply interrupted before its first audio packet never
        reached the boundary's playback path at all, and a turn that got no line is a turn the sweep
        cannot see.
        """
        abandoned = [
            key
            for key, trace in self.trace_log.turns.items()
            if key < (turn, epoch) and not trace.done
        ]
        for key in abandoned:
            self.log.step(
                "turn.cut", key[0], key[1], reason="cancelled before its first packet"
            )
            self._end_turn(*key, status="cut")

    def _record(
        self, point: str, turn: int, epoch: int, ns: int, labels: dict[str, Any]
    ) -> None:
        record = self.trace_log.record(point, turn, epoch, ns, labels)
        if record is not None:
            self.report(record.line())

    async def on_event(self, msg: Msg) -> None:
        if msg.kind is Kind.TRACE:
            payload = msg.json
            self._record(payload["point"], msg.turn, msg.epoch, payload["ns"], payload)
        elif msg.kind is Kind.TOKEN:
            payload = msg.json if msg.body[:1] == b"{" else None
            if payload and "error" in payload:
                self._fail(payload["error"])
            elif payload is None and self.talking:
                self.io.said(msg.text)
        elif msg.kind is Kind.TURN and self.talking:
            self.io.heard(msg.json.get("text", ""), bool(msg.flags & FLAG_FINAL))

    def _fail(self, error: str) -> None:
        self.log.error("reported", error=error)
        if self.on_line is not None:
            self.on_line(f"[{error}]")
        if self.talking:
            self.io.failed(error)

    @property
    def talking(self) -> bool:
        """Whether the console carries the conversation, or only the collected turn lines."""
        return not self.cfg.trace.quiet

    def report(self, line: str) -> None:
        """The turn's numbers, to whoever is collecting and, unless asked not to, to the terminal."""
        self.log.step("reported", line=line)
        if self.on_line is not None:
            self.on_line(line)
        if self.talking:
            self.io.measured(line)
