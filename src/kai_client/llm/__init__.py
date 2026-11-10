from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from kai_client.utils.config import Config


class LlmStream(Protocol):
    """The chat service, seen as a stream of deltas: the composer never learns how they arrive."""

    def stream(
        self, messages: list[dict[str, Any]], *, max_tokens: int | None = None
    ) -> AsyncIterator[str]: ...

    async def prewarm(self) -> None: ...

    async def aclose(self) -> None: ...


def build(cfg: Config) -> LlmStream:
    from kai_client.llm.chat import Chat

    return Chat(cfg.llm)
