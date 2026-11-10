from __future__ import annotations

import logging
import os
import sys
from types import FrameType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kai_client.utils.config import Config

from kai_client.utils import stdio

LEVELS = {
    "off": logging.CRITICAL + 10,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
COLORS = {
    logging.DEBUG: "\x1b[37m",
    logging.INFO: "\x1b[32m",
    logging.WARNING: "\x1b[33m",
    logging.ERROR: "\x1b[31m",
}
RESET = "\x1b[0m"
PACKAGE = "kai_client"
FORMAT = "%(asctime)s - %(name)-28s - %(levelname)-7s - %(message)s"
STAMP = "%H:%M:%S"
WIDTH = 120
_handler: logging.StreamHandler | None = None


class Formatter(logging.Formatter):
    """The origin in place of the logger name, and colour on the log information only.

    Every logger is a child of one root, which is what the level is set on, so the name carries a
    prefix the reader already knows. Dropping it leaves `dirname.module.functioncall`: where the
    line was decided. Colour stops at the message - a transcript or a url that came out red is
    unreadable - and a run piped to a file keeps no escapes at all.
    """

    def __init__(self, tty: bool) -> None:
        super().__init__(FORMAT, STAMP)
        self.tty = tty

    def format(self, record: logging.LogRecord) -> str:
        record.name = record.name.removeprefix(f"{PACKAGE}.")
        text = super().format(record)
        message = record.getMessage()
        colour = COLORS.get(record.levelno, "") if self.tty else ""
        if not colour or not text.endswith(message):
            return text
        return f"{colour}{text[: -len(message)]}{RESET}{message}"


def get_logger(name: str = "") -> logging.Logger:
    return logging.getLogger(f"{PACKAGE}.{name}" if name else PACKAGE)


class _Handler(logging.StreamHandler):
    """A log line that cannot cost the run: the terminal being momentarily full is waited out
    rather than raised through, and the module's own error report would print over the pipeline.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            stdio.write(self.stream, self.format(record) + self.terminator)
        except RecursionError:
            raise
        except Exception:  # noqa: BLE001
            self.handleError(record)


def setup(config: Config) -> None:
    """The one call a process makes: the level the config names, on stderr."""
    setup_logging(str(config.system.log_level), stream=sys.stderr)


def setup_logging(level: str = "warning", stream: Any = None) -> None:
    """One stderr handler for the process, at the level the config names.

    Installed once and re-levelled on every call: a stage process configures this on its way
    through a constructor, and a test wants the shipped default back afterwards.
    """
    global _handler
    target = sys.stderr if stream is None else stream
    root = get_logger()
    if _handler is None:
        _handler = _Handler(target)
        root.addHandler(_handler)
    _handler.stream = target
    _handler.setFormatter(Formatter(_colour(target)))
    root.setLevel(_level(level))
    root.propagate = False


def _level(name: str) -> int:
    try:
        return LEVELS[name.strip().lower()]
    except KeyError:
        raise ValueError(
            f"system.log_level={name!r}, expected one of {', '.join(LEVELS)}"
        ) from None


def _colour(stream: Any) -> bool:
    return "NO_COLOR" not in os.environ and bool(
        getattr(stream, "isatty", lambda: False)()
    )


class Logger:
    """The pipeline's steps, one line each, at the level `config.toml` names.

    `step` is a decision the pipeline can be wrong about - what the gate heard, what the service
    was sent, what came back, what was dropped and why. `detail` is the traffic underneath it,
    which is only legible one frame or packet at a time. Both carry the turn they belong to, and
    every line is attributed to the function that emitted it.
    """

    def __init__(self) -> None:
        if _handler is None:
            setup_logging()

    @property
    def _tracing(self) -> bool:
        """Whether the hot paths are worth formatting at all: False below `detail`'s level."""
        return get_logger().isEnabledFor(logging.DEBUG)

    def step(self, point: str, turn: int = 0, epoch: int = 0, **fields: Any) -> None:
        self._at(logging.INFO, point, turn, epoch, fields)

    def detail(self, point: str, turn: int = 0, epoch: int = 0, **fields: Any) -> None:
        if self._tracing:
            self._at(logging.DEBUG, point, turn, epoch, fields)

    def warning(self, point: str, turn: int = 0, epoch: int = 0, **fields: Any) -> None:
        self._at(logging.WARNING, point, turn, epoch, fields)

    def error(self, point: str, turn: int = 0, epoch: int = 0, **fields: Any) -> None:
        self._at(logging.ERROR, point, turn, epoch, fields)

    def failure(
        self,
        point: str,
        exc: BaseException,
        turn: int = 0,
        epoch: int = 0,
        **fields: Any,
    ) -> None:
        self._log(logging.ERROR, point, turn, epoch, fields, exc, depth=2)

    def _at(
        self, level: int, point: str, turn: int, epoch: int, fields: dict[str, Any]
    ) -> None:
        self._log(level, point, turn, epoch, fields, None, depth=3)

    def _log(
        self,
        level: int,
        point: str,
        turn: int,
        epoch: int,
        fields: dict[str, Any],
        exc: BaseException | None,
        depth: int,
    ) -> None:
        """Attributed to the calling frame, however many of ours it took to get here."""
        logging.getLogger(_origin(sys._getframe(depth))).log(
            level, _line(point, turn, epoch, fields), exc_info=exc
        )


def _origin(frame: FrameType) -> str:
    """`root.dirname.module.functioncall`, the child every line is attributed to.

    Under the package root, because that is where the one handler and the configured level live.
    """
    module = str(frame.f_globals.get("__name__", "")).removeprefix(f"{PACKAGE}.")
    return f"{PACKAGE}.{module}.{frame.f_code.co_name}"


def _line(point: str, turn: int, epoch: int, fields: dict[str, Any]) -> str:
    head = f"turn={turn}.{epoch} " if turn else ""
    return (
        head
        + point
        + "".join(
            f" {key}={_shown(value)}"
            for key, value in fields.items()
            if value is not None
        )
    )


def _shown(value: Any) -> str:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"{len(value)}B"
    text = str(value)
    return text if len(text) <= WIDTH else text[: WIDTH - 3] + "..."
