from __future__ import annotations

import re
from dataclasses import dataclass, field

BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+|[\n\r]+|(?<=[。！？])")
SOFT_BOUNDARY = re.compile(r"(?<=[,;:—–])\s+")
CODE_FENCE = re.compile(r"```.*?(?:```|$)", re.DOTALL)
INLINE_CODE = re.compile(r"`([^`]*)`")
LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
MARKDOWN = re.compile(
    rf"^\s{{0,3}}(#{{1,6}}|[-*+]>?)\s+|\*{{1,3}}|_{{2}}|__|{LINK.pattern}|^\s*[-*+]\s+",
    re.MULTILINE,
)
EMOJI = re.compile(r"[^\w\s.,!?;:'\"()\-/…—，。！？；：、\u00c0-\u024f]", re.UNICODE)


def speakable(text: str) -> str:
    text = CODE_FENCE.sub(" ", text)
    text = INLINE_CODE.sub(r"\1", text)
    text = LINK.sub(r"\1", MARKDOWN.sub(lambda m: m.group(1) or " ", text))
    return EMOJI.sub(" ", text)


@dataclass
class SentenceBatcher:
    first_batch: int = 1
    batch: int = 3
    min_first_chars: int = 20
    max_chunk_chars: int = 180
    _buffer: str = ""
    _sentences: list = field(default_factory=list)
    _emitted: int = 0

    def push(self, text: str) -> list[str]:
        self._buffer += text
        ready: list[str] = []
        while True:
            match = BOUNDARY.search(self._buffer)
            if match is None:
                break
            sentence, self._buffer = (
                self._buffer[: match.start()],
                self._buffer[match.end() :],
            )
            ready.append(sentence)
        if len(self._buffer) > self.max_chunk_chars:
            ready.extend(self._split_long(self._buffer))
            self._buffer = ""
        self._sentences.extend(ready)
        return self._take()

    def finish(self) -> list[str]:
        if self._buffer.strip():
            self._sentences.append(self._buffer)
        self._buffer = ""
        out = self._take()
        if self._sentences:
            out.extend(self._split_long(" ".join(self._sentences)))
            self._sentences = []
        self._emitted += len(out)
        return out

    def _take(self) -> list[str]:
        out: list[str] = []
        while self._sentences:
            if self._emitted == 0:
                want = max(1, self.first_batch)
                if len(self._sentences) < want or self._chars() < self.min_first_chars:
                    break
                count = max(want, self._needed(self.min_first_chars))
            else:
                want = max(1, self.batch)
                if len(self._sentences) < want:
                    break
                count = want
            count = max(1, min(count, self._cover()))
            out.extend(self._split_long(" ".join(self._sentences[:count])))
            self._sentences = self._sentences[count:]
            self._emitted += 1
        return out

    def _chars(self) -> int:
        return len(" ".join(self._sentences).strip())

    def _needed(self, floor: int) -> int:
        total = 0
        for index, sentence in enumerate(self._sentences, 1):
            total += len(sentence) + 1
            if total >= floor:
                return index
        return len(self._sentences)

    def _cover(self) -> int:
        total = 0
        for index, sentence in enumerate(self._sentences):
            total += len(sentence) + 1
            if total > self.max_chunk_chars and index:
                return index
        return len(self._sentences)

    def _split_long(self, text: str) -> list[str]:
        parts: list[str] = []
        while len(text) > self.max_chunk_chars:
            window = text[: self.max_chunk_chars]
            cut = (
                max(m.end() for m in SOFT_BOUNDARY.finditer(window))
                if SOFT_BOUNDARY.search(window)
                else None
            )
            if cut is None:
                cut = text.rfind(" ", 0, self.max_chunk_chars) or self.max_chunk_chars
            parts.append(text[:cut])
            text = text[cut:].lstrip()
        if text:
            parts.append(text)
        return parts
