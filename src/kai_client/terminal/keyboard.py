from __future__ import annotations

import asyncio
import atexit
import contextlib
import io
import os
import sys
import termios
import threading
import tty
from collections.abc import Callable
from time import monotonic
from typing import Any

SPACE = b" "
REPEAT_GAP_S = 0.25


class SpaceKey:
    """The space bar, watched on a thread.

    One callback per physical press: a key held through its auto-repeat, or a burst of spaces that
    arrive in one read, is still one press.

    The read happens off the event loop on purpose. A loop reader (`add_reader`) would put the
    terminal in non-blocking mode, and on macOS that flag belongs to the tty rather than to the
    descriptor, so every other process sharing the terminal would start failing a busy write to
    stdout with `EAGAIN`. A blocking read on a daemon thread costs one parked thread and leaves
    the terminal as it found it.
    """

    def __init__(
        self, on_press: Callable[[], None], *, stream: io.IOBase | None = None
    ) -> None:
        self.on_press = on_press
        self.stream = sys.stdin if stream is None else stream
        self.fd = -1
        self._saved: list[Any] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._quit = threading.Event()
        self._last = 0.0

    @property
    def attached(self) -> bool:
        return self._saved is not None

    def start(self) -> bool:
        if self.attached:
            return True
        try:
            fd = self.stream.fileno()
            saved = termios.tcgetattr(fd)
        except OSError, ValueError, termios.error:
            return False
        self.fd, self._saved = fd, saved
        self._loop = asyncio.get_running_loop()
        tty.setcbreak(fd)
        self._last = 0.0
        self._quit.clear()
        self._thread = threading.Thread(
            target=self._watch, name="space-key", daemon=True
        )
        self._thread.start()
        atexit.register(self.stop)
        return True

    def stop(self) -> None:
        saved, self._saved = self._saved, None
        self._quit.set()
        if saved is None:
            return
        with contextlib.suppress(Exception):
            termios.tcsetattr(self.fd, termios.TCSADRAIN, saved)

    def _watch(self) -> None:
        while not self._quit.is_set():
            try:
                typed = os.read(self.fd, 64)
            except OSError, ValueError, KeyboardInterrupt:
                return
            if not typed or self._quit.is_set():
                return
            if SPACE in typed and self._fresh():
                self._deliver()

    def _fresh(self) -> bool:
        gap, self._last = monotonic() - self._last, monotonic()
        return gap >= REPEAT_GAP_S

    def _deliver(self) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(self.on_press)
