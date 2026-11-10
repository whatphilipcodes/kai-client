from __future__ import annotations

import contextlib
import os

import zmq
import zmq.asyncio

from kai_client.runtime.protocol import Frame
from kai_client.utils.config import Endpoint, Transport

CONTEXT: zmq.asyncio.Context | None = None

SENDERS = (zmq.PUSH, zmq.PUB)


def context() -> zmq.asyncio.Context:
    global CONTEXT
    if CONTEXT is None:
        CONTEXT = zmq.asyncio.Context()
    return CONTEXT


def _bind(socket: zmq.Socket, endpoint: Endpoint) -> None:
    if endpoint.protocol is Transport.IPC:
        with contextlib.suppress(OSError):
            os.unlink(endpoint.path)
    socket.bind(endpoint.address)


class Pipe:
    """One zmq socket, either the speaking end or the listening end of an address."""

    def __init__(self, kind: int, endpoints: list[Endpoint], *, hwm: int) -> None:
        self.dropped = 0
        self.endpoints = list(endpoints)
        self.socket = context().socket(kind)
        self.sends = kind in SENDERS
        self.socket.setsockopt(zmq.SNDHWM if self.sends else zmq.RCVHWM, hwm)
        for endpoint in self.endpoints:
            _bind(self.socket, endpoint) if self.sends else self.socket.connect(
                endpoint.address
            )

    def send(self, frames: list[Frame]) -> None:
        try:
            self.socket.send_multipart(frames, flags=zmq.NOBLOCK, copy=False)
        except zmq.Again:
            self.dropped += 1

    async def recv(self) -> list[zmq.Frame]:
        return await self.socket.recv_multipart(copy=False)

    def close(self) -> None:
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.close()


class Stream(Pipe):
    """PUSH/PULL over one address: the token plane, where a full queue must not stall a stage."""

    def __init__(self, endpoint: Endpoint, *, bind: bool, hwm: int = 1000) -> None:
        self.endpoint = endpoint
        super().__init__(zmq.PUSH if bind else zmq.PULL, [endpoint], hwm=hwm)


class Events(Pipe):
    """PUB/SUB: the state plane, one address to speak on and many to listen to."""

    def __init__(
        self,
        endpoints: Endpoint | list[Endpoint],
        *,
        bind: bool,
        topics: tuple[bytes, ...] = (b"",),
    ) -> None:
        peers = [endpoints] if isinstance(endpoints, Endpoint) else list(endpoints)
        super().__init__(zmq.PUB if bind else zmq.SUB, peers, hwm=100 if bind else 1000)
        if not bind:
            for topic in topics:
                self.socket.setsockopt(zmq.SUBSCRIBE, topic)
