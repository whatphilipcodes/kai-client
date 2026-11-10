from __future__ import annotations

import sys
from typing import TextIO

from kai_client.utils import stdio

COLUMN = 7
CLEAR = "\x1b[K"


class Dialogue:
    """The exchange as text: what the microphone said is `human`, what came back is `kai`.

    A transcript that is still being revised rewrites its own line rather than printing another one,
    and an identical revision is dropped, so a growing sentence stays one line. What the turn cost
    goes under it.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = sys.stdout if stream is None else stream
        self.open = ""
        self.partial = ""

    @property
    def tty(self) -> bool:
        return bool(self.stream.isatty())

    def heard(self, text: str, final: bool) -> None:
        """The words the ASR made of the turn.

        A partial rewrites its own line, which is what a growing transcript looks like from the
        inside; in a log a partial is not printed at all, because a pipe cannot unprint a sentence.
        """
        text = text.strip()
        if (
            not text
            or (text == self.partial and not final)
            or (not final and not self.tty)
        ):
            return
        rewriting = self.open == "human"
        if not rewriting:
            self.close()
        start = "\r" if rewriting else ""
        end = "\n" if final else CLEAR
        stdio.write(self.stream, f"{start}{'human':<{COLUMN}}{text}{end}")
        self.open, self.partial = ("" if final else "human"), ("" if final else text)

    def said(self, text: str) -> None:
        if self.open != "kai":
            self.close()
            stdio.write(self.stream, f"{'kai':<{COLUMN}}")
            self.open = "kai"
            text = text.lstrip()
        stdio.write(self.stream, text)

    def failed(self, error: str) -> None:
        self.close()
        stdio.write(self.stream, f"{'kai':<{COLUMN}}[{error}]\n")

    def measured(self, line: str) -> None:
        self.close()
        stdio.write(self.stream, f"{'':<{COLUMN}}[{line}]\n\n")

    def close(self) -> None:
        if self.open:
            stdio.write(self.stream, "\n")
        self.open = self.partial = ""
