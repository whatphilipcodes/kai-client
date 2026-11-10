from __future__ import annotations

import asyncio
from hashlib import blake2b
from time import monotonic_ns
from typing import Any

from kai_client.asr import AsrService, AsrSession
from kai_client.asr.vad import (
    Event,
    Gate,
    ManualVad,
    SpeechEnd,
    SpeechPending,
    SpeechStart,
)
from kai_client.asr.vad import build as build_vad
from kai_client.audio import pcm
from kai_client.runtime import wire
from kai_client.runtime.bus import Stream
from kai_client.runtime.protocol import (
    FLAG_AUDIO,
    FLAG_FINAL,
    FLAG_PARTIAL,
    Kind,
    encode_info,
)
from kai_client.runtime.stage import Stage
from kai_client.terminal import keyboard
from kai_client.utils.config import Config

STALL_GRACE_NS = 200_000_000


class Transcriber(Stage):
    name = "asr"

    def __init__(self, cfg: Config, asr: AsrService | None = None) -> None:
        super().__init__(cfg)
        self.audio_in = wire.UdpReceiver(cfg.network.udp("mic"))
        self.turns = Stream(cfg.network.stream("turn"), bind=True)
        self.frame_bytes = pcm.frame_bytes(cfg.audio.frame_ms, cfg.audio.input_rate)
        self.vad: Gate = build_vad(cfg.vad, cfg.audio.input_rate, self.frame_bytes)
        self.keys = keyboard.SpaceKey(self._on_press)
        self.asr = asr
        self.backends = (asr,)
        self.session: AsrSession | None = None
        self.turn = 0
        self.epoch = 0
        self.revision = 0
        self._snapshot_task: asyncio.Task | None = None
        self._timer_task: asyncio.Task | None = None
        self._final_task: asyncio.Task | None = None
        self._busy = False
        self._last_ns = 0

    async def start(self) -> None:
        await super().start()
        self.audio_in.attach(asyncio.get_running_loop(), self._on_packet)
        self.log.step(
            "gate.load",
            engine=type(self.vad).__name__,
            frame_bytes=self.frame_bytes,
            mic_port=self.audio_in.endpoint.port,
        )
        if isinstance(self.vad, ManualVad):
            if self.keys.start():
                self.log.step("gate.key", keys="space opens and closes the gate")
            else:
                self.log.warning(
                    "gate.key", reason="stdin is no terminal: nothing can open the gate"
                )
        self.tasks.append(asyncio.create_task(self._watch_stall(), name="asr.stall"))

    async def stop(self) -> None:
        self.keys.stop()
        self._session_off()
        self.audio_in.close()
        for task in (self._snapshot_task, self._timer_task, self._final_task):
            if task is not None:
                task.cancel()
        await super().stop()

    def state(self) -> dict[str, Any]:
        return {"turn": self.turn, "epoch": self.epoch, "listening": self.vad.speech}

    async def _watch_stall(self) -> None:
        stall_ns = int(self.cfg.vad.min_silence_ms * 1e6) + STALL_GRACE_NS
        while not self.stopping.is_set():
            await asyncio.sleep(0.05)
            if self.vad.speech and monotonic_ns() - self._last_ns > stall_ns:
                end = self.vad.flush()
                if end is not None:
                    self.log.warning(
                        "mic.stalled",
                        turn=self.turn,
                        epoch=self.epoch,
                        voiced_ms=round(end.duration_ms, 1),
                    )
                    self._close(end)

    def _on_packet(self, packet: wire.Packet) -> None:
        if packet.stream != wire.MIC:
            return
        self._last_ns = packet.sent_ns
        self.log.detail(
            "mic.packet",
            self.turn,
            self.epoch,
            seq=packet.seq,
            bytes=len(packet.pcm),
            rate=packet.rate,
        )
        self._route(self.vad.feed(bytes(packet.pcm), packet.sent_ns))
        if self.vad.speech and self.session is not None:
            self.session.push(self.vad.voiced_pcm())

    def _on_press(self) -> None:
        if isinstance(self.vad, ManualVad):
            self._route(self.vad.toggle())

    def _route(self, events: list[Event]) -> None:
        for event in events:
            match event:
                case SpeechStart():
                    self._open(event)
                case SpeechPending():
                    self._warm()
                case SpeechEnd():
                    self._close(event)

    def _open(self, event: SpeechStart) -> None:
        if self.peer_speaking():
            self.epoch += 1
            self.cancel(event.turn, self.epoch, "barge_in")
        self._stop_capture_tasks()
        if self._final_task is not None and not self._final_task.done():
            self.log.warning(
                "turn.abandoned",
                turn=self.turn,
                epoch=self.epoch,
                reason="the speaker started again",
            )
            self._final_task.cancel()
        self.turn = event.turn
        self.revision = 0
        self._busy = False
        self._session_off()
        pre_roll = self.vad.voiced_pcm()
        self.log.step(
            "vad.open",
            self.turn,
            self.epoch,
            min_speech_ms=self.cfg.vad.min_speech_ms,
            rms=self.cfg.vad.rms_open,
        )
        if self.asr is not None:
            self.session = self.asr.session()
            self.session.push(pre_roll)
            self.log.step("session.open", self.turn, self.epoch, pre_roll=len(pre_roll))
        else:
            self.log.step("session.skip", self.turn, self.epoch, mode=self.cfg.asr.mode)
        self.tracer.mark("turn_start", self.turn, self.epoch)
        self._timer_task = asyncio.create_task(self._partials(), name="asr.partials")

    def _close(self, event: SpeechEnd) -> None:
        self._stop_capture_tasks()
        if event.turn != self.turn:
            self.log.detail("gate.late", event.turn, self.epoch, seen=self.turn)
            return
        self.log.step(
            "vad.close",
            self.turn,
            self.epoch,
            voiced_ms=round(event.duration_ms, 1),
            silence_ms=round(event.trailing_silence_ms, 1),
            bytes=len(event.pcm),
        )
        self.tracer.mark("mic_end", self.turn, self.epoch, ns=event.voiced_end_ns)
        self.tracer.mark("turn_open", self.turn, self.epoch)
        self.tracer.mark("stt_start", self.turn, self.epoch)
        self._final_task = asyncio.create_task(self._final(event), name="asr-final")

    def _session_off(self) -> None:
        if self.session is not None:
            self.log.detail("session.close", self.turn, self.epoch)
            self.session.abort()
            self.session = None

    def _stop_capture_tasks(self) -> None:
        current = asyncio.current_task()
        for task in (self._snapshot_task, self._timer_task):
            if task is not None and task is not current:
                task.cancel()
        self._snapshot_task = self._timer_task = None

    async def _partials(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.vad.partial_every_ms / 1000)
            self._warm()

    def _warm(self) -> None:
        if self._busy:
            return
        self._snapshot_task = asyncio.create_task(self._snapshot(), name="asr-snapshot")

    async def _snapshot(self) -> None:
        voiced, duration_ms = self.vad.voiced_pcm(), self.vad.voiced_ms()
        if self.session is None:
            self._segment(voiced, duration_ms, final=False)
            return
        turn, epoch, revision = self.turn, self.epoch, self.revision + 1
        self._busy = True
        self.session.push(voiced)
        try:
            text = await self.session.text()
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001
            if not self.session.unreachable:
                self.log.warning(
                    "partial.failed",
                    turn=turn,
                    epoch=epoch,
                    error=f"{type(exc).__name__}: {exc}",
                )
            end = self.vad.flush()
            if end is not None:
                self._close(end)
            return
        finally:
            self._busy = False
        if not text.strip() or (turn, epoch) != (self.turn, self.epoch):
            self.log.detail(
                "partial.dropped", turn, epoch, text=text, current=self.turn
            )
            return
        self._spoken_text(turn, epoch, revision, text, duration_ms, final=False)

    async def _final(self, event: SpeechEnd) -> None:
        turn, epoch, revision, duration_ms = (
            self.turn,
            self.epoch,
            self.revision + 1,
            event.duration_ms,
        )
        if self.session is None:
            self._segment(event.pcm[: event.voiced_len], duration_ms, final=True)
            return
        self.session.push(event.pcm)
        began = monotonic_ns()
        try:
            text = await self.session.finish()
        except asyncio.CancelledError:
            self.log.warning(
                "turn.cancelled",
                turn=turn,
                epoch=epoch,
                reason="a new turn opened before the answer landed",
            )
            return
        except Exception as exc:  # noqa: BLE001
            if self.session.unreachable:
                self.log.warning(
                    "stt.unreachable", turn=turn, epoch=epoch, url=self.cfg.asr.url
                )
                return
            self.log.error(
                "stt.failed",
                turn=turn,
                epoch=epoch,
                error=f"{type(exc).__name__}: {exc}",
            )
            self.turns.send(
                encode_info(
                    Kind.TURN,
                    turn,
                    epoch,
                    {"error": str(exc)},
                    seq=revision,
                    flags=FLAG_FINAL,
                )
            )
            return
        self.log.step(
            "stt.final",
            turn,
            epoch,
            ms=round((monotonic_ns() - began) / 1e6, 1),
            text=text,
        )
        self._spoken_text(turn, epoch, revision, text, duration_ms, final=True)

    def _spoken_text(
        self,
        turn: int,
        epoch: int,
        revision: int,
        text: str,
        duration: float,
        final: bool,
    ) -> None:
        payload = {"text": text, "duration_ms": duration, "final": final}
        self._send(
            turn, epoch, revision, payload, FLAG_FINAL if final else FLAG_PARTIAL
        )

    def _send(
        self,
        turn: int,
        epoch: int,
        revision: int,
        payload: dict[str, Any],
        flags: int,
        audio: bytes = b"",
    ) -> None:
        if flags & FLAG_FINAL:
            self.tracer.mark("stt_end", turn, epoch)
        line = self.log.step if flags & FLAG_FINAL else self.log.detail
        line(
            "turn.publish",
            turn,
            epoch,
            revision=revision,
            final=bool(flags & FLAG_FINAL),
            partial=bool(flags & FLAG_PARTIAL),
            audio=bool(flags & FLAG_AUDIO),
            bytes=len(audio),
        )
        frames = encode_info(
            Kind.TURN, turn, epoch, payload, seq=revision, flags=flags, audio=audio
        )
        self.turns.send(frames)
        if not flags & FLAG_AUDIO:
            self.events_out.send(frames)
        self.revision = revision

    def _segment(self, audio: bytes, duration: float, final: bool) -> None:
        turn, epoch, revision = self.turn, self.epoch, self.revision + 1
        payload = {
            "audio_id": blake2b(audio, digest_size=8).hexdigest(),
            "rate": self.cfg.audio.input_rate,
            "duration_ms": duration,
            "final": final,
        }
        self._send(
            turn,
            epoch,
            revision,
            payload,
            FLAG_AUDIO | (FLAG_FINAL if final else FLAG_PARTIAL),
            audio,
        )
