from __future__ import annotations

import asyncio
import contextlib
from time import monotonic_ns
from typing import Any

from kai_client.runtime.bus import Events
from kai_client.runtime.protocol import EpochGuard, Kind, Msg, decode, encode_info
from kai_client.runtime.trace import Tracer
from kai_client.utils.config import STAGES, Config
from kai_client.utils.logger import Logger

PEER_STATE_TTL_NS = 2_000_000_000


class Stage:
    name = ""
    event_topics: tuple[Kind, ...] = (Kind.HELLO, Kind.STATE, Kind.CANCEL)

    def __init__(self, cfg: Config) -> None:
        cfg.prepare()
        self.cfg = cfg
        self.log = Logger()
        self.addresses = [
            cfg.network.events(other) for other in STAGES if other != self.name
        ]
        self.events_out = Events(cfg.network.events(self.name), bind=True)
        self.events_in = Events(
            self.addresses,
            bind=False,
            topics=tuple(bytes([int(k)]) for k in self.event_topics),
        )
        self.guard = EpochGuard()
        self.tracer = Tracer(self.events_out, cfg.trace.enabled)
        self.peers: dict[str, Msg] = {}
        self.tasks: list[asyncio.Task] = []
        self.backends: tuple[Any, ...] = ()
        self.stopping = asyncio.Event()

    async def start(self) -> None:
        self.log.step(
            "stage.start",
            bind=self.events_out.endpoints[0].address,
            peers=len(self.addresses),
        )
        self.tasks.append(
            asyncio.create_task(self._pump_events(), name=f"{self.name}.events")
        )
        self.tasks.append(
            asyncio.create_task(self._heartbeat(), name=f"{self.name}.heartbeat")
        )

    async def stop(self) -> None:
        self.stopping.set()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        for backend in self.backends:
            if backend is not None:
                await backend.aclose()
        self.log.step("stage.stop", dropped=self.events_in.dropped)
        self.events_out.close()
        self.events_in.close()

    async def _pump_events(self) -> None:
        while not self.stopping.is_set():
            try:
                frames = await self.events_in.recv()
            except asyncio.CancelledError:
                return
            msg = decode(frames)
            if msg.kind not in (Kind.HELLO, Kind.STATE):
                self.log.detail(
                    "event.receive",
                    msg.turn,
                    msg.epoch,
                    kind=msg.kind.name,
                    body=msg.text[:80],
                )
            try:
                if msg.kind is Kind.CANCEL:
                    self.on_cancel(msg)
                elif msg.kind in (Kind.HELLO, Kind.STATE):
                    self.peers[msg.json.get("stage", "")] = msg
                await self.on_event(msg)
            except Exception as exc:  # noqa: BLE001
                self.log.failure(
                    "event.dropped", exc, msg.turn, msg.epoch, kind=msg.kind.name
                )

    async def _heartbeat(self) -> None:
        announced: dict[str, Any] = {}
        while not self.stopping.is_set():
            state: dict[str, Any] = {"stage": self.name, **self.state()}
            if state != announced:
                self.log.detail("state", **state)
                announced = state
            self.announce(state)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.stopping.wait(), timeout=0.25)

    def announce(self, state: dict[str, Any]) -> None:
        self.events_out.send(encode_info(Kind.HELLO, 0, 0, state))

    def state(self) -> dict[str, Any]:
        return {}

    def peer_speaking(self) -> bool:
        return self._peer_flag("tts", "speaking") or self._peer_flag(
            "llm", "generating"
        )

    def _peer_flag(self, stage: str, key: str) -> bool:
        peer = self.peers.get(stage)
        if peer is None:
            return False
        return (
            bool(peer.json.get(key))
            and monotonic_ns() - peer.sent_ns < PEER_STATE_TTL_NS
        )

    def cancel(self, turn: int, epoch: int, reason: str) -> None:
        self.log.step("cancel", turn, epoch, by=self.name, reason=reason)
        self.events_out.send(
            encode_info(
                Kind.CANCEL, turn, epoch, {"stage": self.name, "reason": reason}
            )
        )

    def on_cancel(self, msg: Msg) -> None:
        pass

    async def on_event(self, msg: Msg) -> None:
        pass
