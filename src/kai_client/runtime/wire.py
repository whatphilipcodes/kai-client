from __future__ import annotations

import asyncio
import contextlib
import socket
import struct
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic_ns

from kai_client.utils.config import Endpoint

HEADER = struct.Struct("<BBBBIIIIQ")
HEADER_BYTES = HEADER.size

FIRST = 0b001
LAST = 0b010
SILENCE = 0b100

MIC = 1
TTS = 2


@dataclass(slots=True)
class Packet:
    stream: int
    turn: int
    epoch: int
    seq: int
    rate: int
    flags: int
    sent_ns: int
    pcm: memoryview


def encode(
    stream: int, turn: int, epoch: int, seq: int, rate: int, pcm: bytes, flags: int = 0
) -> bytes:
    return (
        HEADER.pack(1, stream, flags, 0, turn, epoch, seq, rate, monotonic_ns()) + pcm
    )


def decode(data: bytes | bytearray) -> Packet | None:
    if len(data) < HEADER_BYTES:
        return None
    _, stream, flags, _, turn, epoch, seq, rate, sent_ns = HEADER.unpack_from(data)
    return Packet(
        stream, turn, epoch, seq, rate, flags, sent_ns, memoryview(data)[HEADER_BYTES:]
    )


class UdpSender:
    def __init__(self, endpoint: Endpoint) -> None:
        self.endpoint = endpoint
        self._addr = (str(endpoint.host), endpoint.port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        self._sock.setblocking(False)

    def send(self, datagram: bytes) -> None:
        with contextlib.suppress(OSError):
            self._sock.sendto(datagram, self._addr)

    def close(self) -> None:
        self._sock.close()


class UdpReceiver:
    TRACKED = 32

    def __init__(self, endpoint: Endpoint, depth: int = 1 << 18) -> None:
        self.endpoint = endpoint
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, depth)
        self._sock.bind((str(endpoint.host), endpoint.port))
        self._sock.setblocking(False)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_seq: dict[tuple[int, int], int] = {}
        self.malformed = 0
        self.gaps = 0

    @property
    def port(self) -> int:
        return self._sock.getsockname()[1]

    def attach(
        self, loop: asyncio.AbstractEventLoop, callback: Callable[[Packet], None]
    ) -> None:
        self._loop = loop
        loop.add_reader(self._sock.fileno(), self._drain, callback)

    def detach(self) -> None:
        if self._loop is not None:
            self._loop.remove_reader(self._sock.fileno())
            self._loop = None

    def _drain(self, callback: Callable[[Packet], None]) -> None:
        for _ in range(256):
            try:
                data = self._sock.recv(4096)
            except BlockingIOError, OSError:
                return
            if not data:
                return
            packet = decode(data)
            if packet is None:
                self.malformed += 1
                continue
            if packet.flags & SILENCE:
                callback(packet)
                continue
            turn = (packet.stream, packet.turn)
            previous = self._last_seq.get(turn)
            if previous is not None and packet.seq > previous + 1:
                self.gaps += 1
            if len(self._last_seq) > self.TRACKED and turn not in self._last_seq:
                del self._last_seq[next(iter(self._last_seq))]
            self._last_seq[turn] = packet.seq
            callback(packet)

    def close(self) -> None:
        self.detach()
        with contextlib.suppress(OSError):
            self._sock.close()
