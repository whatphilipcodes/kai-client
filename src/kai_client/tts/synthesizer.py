from __future__ import annotations

import asyncio
from typing import Any

from kai_client.audio import pcm
from kai_client.runtime import wire
from kai_client.runtime.bus import Stream
from kai_client.runtime.protocol import FLAG_FINAL, Kind, Msg, decode, encode_info
from kai_client.runtime.stage import Stage
from kai_client.tts import TtsStream
from kai_client.utils.config import Config


class Synthesizer(Stage):
    name = "tts"

    def __init__(self, cfg: Config, tts: TtsStream) -> None:
        super().__init__(cfg)
        self.sentences_in = Stream(cfg.network.stream("sentence"), bind=False)
        self.audio_out = wire.UdpSender(cfg.network.udp("tts"))
        self.tts = tts
        self.backends = (tts,)
        self.rate = tts.sample_rate
        self.frame_bytes = pcm.frame_bytes(cfg.audio.frame_ms, self.rate)
        self.seq = 0
        self.speaking = False

    async def start(self) -> None:
        await super().start()
        await self.tts.register()
        self.tasks.append(
            asyncio.create_task(self._pump_sentences(), name="tts.sentences")
        )

    def state(self) -> dict[str, Any]:
        return {"speaking": self.speaking, "rate": self.rate}

    def on_cancel(self, msg: Msg) -> None:
        if self.guard.adopt(msg.turn, msg.epoch):
            self.speaking = False

    async def _pump_sentences(self) -> None:
        while not self.stopping.is_set():
            try:
                frames = await self.sentences_in.recv()
            except asyncio.CancelledError:
                return
            msg = decode(frames)
            if msg.kind is not Kind.SENTENCE or not self.guard.accept(msg):
                self.log.detail("sentence.stale", msg.turn, msg.epoch, seq=msg.seq)
                continue
            if msg.flags & FLAG_FINAL:
                self._close_stream(msg)
            elif msg.body:
                self.log.detail(
                    "sentence.receive", msg.turn, msg.epoch, n=msg.seq, text=msg.text
                )
                await self._synthesize(msg)

    def _close_stream(self, msg: Msg) -> None:
        self.tracer.mark("audio_end", msg.turn, msg.epoch)
        self.seq += 1
        self.audio_out.send(
            wire.encode(
                wire.TTS, msg.turn, msg.epoch, self.seq, self.rate, b"", wire.LAST
            )
        )
        self.log.step("audio.last", msg.turn, msg.epoch, packets=self.seq)
        self.speaking = False

    async def _synthesize(self, msg: Msg) -> None:
        self.speaking = True
        self._adopt_rate(self.tts.sample_rate)
        self.tracer.mark("tts_start", msg.turn, msg.epoch)
        first = True
        pending = b""
        try:
            async for block in self.tts.stream(msg.text):
                if self.guard.stale(msg.turn, msg.epoch):
                    self.tracer.mark("tts_stopped", msg.turn, msg.epoch)
                    self.log.step(
                        "audio.cancelled",
                        msg.turn,
                        msg.epoch,
                        now=self.guard.turn,
                        packets=self.seq,
                    )
                    break
                pending += block
                while len(pending) >= self.frame_bytes:
                    frame, pending = (
                        pending[: self.frame_bytes],
                        pending[self.frame_bytes :],
                    )
                    flags = wire.FIRST if first else 0
                    self._send(msg, frame, flags)
                    if first:
                        first = False
                        self.tracer.mark("tts_first", msg.turn, msg.epoch)
                        self.log.step(
                            "audio.first", msg.turn, msg.epoch, bytes=len(frame)
                        )
            if pending and not first:
                self._send(msg, pending, 0)
            elif pending:
                self.log.detail("audio.tail", msg.turn, msg.epoch, bytes=len(pending))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.log.failure("tts.failed", exc, msg.turn, msg.epoch, text=msg.text)
            self.events_out.send(
                encode_info(
                    Kind.TOKEN,
                    msg.turn,
                    msg.epoch,
                    {"error": f"tts {type(exc).__name__}: {exc}"},
                )
            )
        finally:
            self.speaking = False

    def _adopt_rate(self, rate: int) -> None:
        if rate == self.rate:
            return
        self.log.step("audio.rate", from_rate=self.rate, to_rate=rate)
        self.rate = rate
        self.frame_bytes = pcm.frame_bytes(self.cfg.audio.frame_ms, rate)

    def _send(self, msg: Msg, audio: bytes, flags: int) -> None:
        self.seq += 1
        self.log.detail(
            "audio.packet",
            msg.turn,
            msg.epoch,
            seq=self.seq,
            bytes=len(audio),
            flags=flags,
        )
        self.audio_out.send(
            wire.encode(
                wire.TTS, msg.turn, msg.epoch, self.seq, self.rate, audio, flags
            )
        )
