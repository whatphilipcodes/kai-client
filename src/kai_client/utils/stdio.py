from __future__ import annotations

import contextlib
import os
import select
import sys
from time import monotonic
from typing import TextIO

GIVE_UP_S = 0.1
WAIT_S = 0.01


def write(stream: TextIO, text: str) -> None:
    """One write to a terminal that may be non-blocking, which is not the writer's choice.

    `O_NONBLOCK` lives on the tty device rather than on the descriptor, so any process attached
    to the terminal — including an event loop watching a key — makes every other process on it
    fail a busy write with `EAGAIN`. Waiting a moment for the device to take the bytes is what a
    blocking write would have done; giving up is what keeps the audio loop running. A line lost
    from the console is still in the log, and a raise here would cost the turn report after it.
    """
    if sys.platform == "win32":
        with contextlib.suppress(OSError, ValueError):
            stream.write(text)
            stream.flush()
        return
    try:
        fd = stream.fileno()
    except AttributeError, OSError, ValueError:
        stream.write(text)
        return
    with contextlib.suppress(BlockingIOError, OSError, ValueError):
        stream.flush()
    data = text.encode(getattr(stream, "encoding", None) or "utf-8", "backslashreplace")
    deadline = monotonic() + GIVE_UP_S
    while data:
        try:
            written = os.write(fd, data)
        except BlockingIOError:
            if monotonic() > deadline:
                return
            select.select([], [fd], [], WAIT_S)
            continue
        except OSError:
            return
        if written <= 0:
            return
        data = data[written:]
