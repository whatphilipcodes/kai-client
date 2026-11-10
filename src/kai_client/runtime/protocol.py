from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from struct import Struct
from time import monotonic_ns
from typing import Any

import zmq

HEADER = Struct("<BBHIIIQ")
Frame = bytes | memoryview | zmq.Frame


class Kind(IntEnum):
    HELLO = 0
    TURN = 1
    SENTENCE = 2
    TOKEN = 3
    INFO = 4
    CANCEL = 5
    TRACE = 6
    STATE = 7


FLAG_FINAL = 0b0001
FLAG_PARTIAL = 0b0010
FLAG_FIRST = 0b0100
FLAG_PREFETCH = 0b1000
FLAG_AUDIO = 0b10000


@dataclass(slots=True)
class Msg:
    kind: Kind
    turn: int
    epoch: int
    seq: int
    flags: int
    sent_ns: int
    body: bytes
    audio: bytes = b""

    @property
    def text(self) -> str:
        return self.body.decode("utf-8")

    @property
    def json(self) -> dict[str, Any]:
        return json.loads(self.body)


def encode(
    kind: Kind,
    turn: int,
    epoch: int,
    seq: int = 0,
    body: bytes = b"",
    flags: int = 0,
    audio: bytes = b"",
) -> list[Frame]:
    frames: list[Frame] = [
        bytes([int(kind)]),
        HEADER.pack(1, flags, 0, turn, epoch, seq, monotonic_ns()),
        body,
    ]
    if audio:
        frames.append(audio)
    return frames


def encode_info(
    kind: Kind,
    turn: int,
    epoch: int,
    payload: dict[str, Any],
    seq: int = 0,
    flags: int = 0,
    audio: bytes = b"",
) -> list[Frame]:
    return encode(
        kind,
        turn,
        epoch,
        seq,
        json.dumps(payload, separators=(",", ":")).encode(),
        flags,
        audio,
    )


def decode(frames: Sequence[Frame]) -> Msg:
    kind = _bytes(frames[0])[0]
    header = _bytes(frames[1])
    body = _bytes(frames[2])
    audio = _bytes(frames[3]) if len(frames) > 3 else b""
    _, flags, _, turn, epoch, seq, sent_ns = HEADER.unpack(header)
    return Msg(Kind(kind), turn, epoch, seq, flags, sent_ns, body, audio)


def _bytes(frame: Frame) -> bytes:
    if isinstance(frame, zmq.Frame):
        return frame.bytes
    return bytes(frame)


class EpochGuard:
    def __init__(self) -> None:
        self.turn = 0
        self.epoch = 0
        self.dropped = 0

    def stale(self, turn: int, epoch: int) -> bool:
        return (turn, epoch) < (self.turn, self.epoch)

    def accept(self, msg: Msg) -> bool:
        if self.stale(msg.turn, msg.epoch):
            self.dropped += 1
            return False
        self.turn, self.epoch = msg.turn, msg.epoch
        return True

    def adopt(self, turn: int, epoch: int) -> bool:
        if self.stale(turn, epoch):
            self.dropped += 1
            return False
        self.turn, self.epoch = turn, epoch
        return True
