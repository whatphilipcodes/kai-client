from __future__ import annotations

from typing import Protocol

from kai_client.utils.config import AsrMode, Config


class AsrSession(Protocol):
    """One utterance, seen by the transcription service as it grows.

    `push` is given the turn's audio so far, not the newest frame: the realtime backend sends what
    it has not sent yet and the batch backend simply re-posts it, so the transcriber has one shape
    to drive whichever service the mode names.
    """

    def push(self, pcm: bytes) -> None: ...

    async def text(self) -> str: ...

    async def finish(self) -> str: ...

    def abort(self) -> None: ...

    @property
    def unreachable(self) -> bool:
        """The service was never reached: nothing is listening at that address.

        A turn that failed because the model said something odd is worth reporting to the
        conversation; a turn that failed because there is no model is worth reporting once.
        """
        ...


class AsrService(Protocol):
    """Something that can be asked to transcribe a turn, and closed when the stage stops."""

    def session(self) -> AsrSession: ...

    async def aclose(self) -> None: ...


def build(cfg: Config) -> AsrService | None:
    """`direct` is the audio-input mode: the turn's speech goes to the LLM unheard by any ASR."""
    if cfg.asr.mode is AsrMode.DIRECT:
        return None
    if cfg.asr.mode is AsrMode.REALTIME:
        from kai_client.asr.realtime import Realtime

        return Realtime(cfg.asr, cfg.audio.input_rate)
    from kai_client.asr.batch import Batch

    return Batch(cfg.asr, cfg.audio.input_rate)
