from __future__ import annotations

import asyncio
import contextlib
import signal
from collections.abc import Callable

from kai_client import asr, llm, tts
from kai_client.audio import devices
from kai_client.runtime.stage import Stage
from kai_client.utils import logger
from kai_client.utils.config import STAGES, Config


def stage(
    cfg: Config,
    name: str,
    *,
    mic: devices.Capture | None = None,
    speaker: devices.Playback | None = None,
    on_line: Callable[[str], None] | None = None,
) -> Stage:
    """The one place a stage name becomes an object, with the import inside its own branch: a
    stage process loads the modality it runs and nothing else, which keeps PortAudio out of the
    three that never touch a device and shortens how long a started process is not yet killable.
    """
    if name == "audio":
        from kai_client.audio.boundary import Boundary

        return Boundary(cfg, mic=mic, speaker=speaker, on_line=on_line)
    if name == "asr":
        from kai_client.asr.transcriber import Transcriber

        return Transcriber(cfg, asr.build(cfg))
    if name == "llm":
        from kai_client.llm.composer import Composer

        return Composer(cfg, llm.build(cfg))
    if name == "tts":
        from kai_client.tts.synthesizer import Synthesizer

        return Synthesizer(cfg, tts.build(cfg))
    raise ValueError(f"unknown stage {name!r}, expected one of {', '.join(STAGES)}")


async def run(
    cfg: Config,
    *,
    stages: tuple[str, ...] = STAGES,
    turns: int = 1,
    timeout: float = 30.0,
    mic: devices.Capture | None = None,
    speaker: devices.Playback | None = None,
    on_event: Callable[[str], None] | None = None,
) -> list[str]:
    """Every stage in one loop: the same endpoints and code paths, no cross-process hop."""
    cfg.prepare()
    lines: list[str] = []

    def report(line: str) -> None:
        lines.append(line)
        if on_event is not None:
            on_event(line)

    logger.setup(cfg)
    built = [
        stage(cfg, name, mic=mic, speaker=speaker, on_line=report) for name in stages
    ]
    await asyncio.gather(*(built_stage.start() for built_stage in built))
    await asyncio.sleep(cfg.network.connect_grace_ms / 1000)
    try:
        await _wait_for(lines, turns, timeout)
    finally:
        for built_stage in reversed(built):
            await built_stage.stop()
    return lines


async def serve(cfg: Config, name: str) -> None:
    """One stage, alone, until the terminal stops it. This is what a launched process runs."""
    cfg.prepare()
    logger.setup(cfg)
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(signal_number, stopping.set)
    built = stage(cfg, name)
    await built.start()
    try:
        await stopping.wait()
    finally:
        await built.stop()


async def _wait_for(lines: list[str], turns: int, timeout: float) -> None:
    if not turns:
        await asyncio.Event().wait()
        return
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if len(lines) >= turns:
            return
        await asyncio.sleep(0.02)
