from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from kai_client.utils.config import Config, dump
from kai_client.utils.logger import Logger

WORKERS = ("tts", "llm", "asr")


def workers(cfg: Config) -> list[subprocess.Popen]:
    """Every stage but the microphone, one interpreter each, bound and connected."""
    return _launch(cfg, WORKERS)


def session(cfg: Config) -> int:
    """The command: four stage processes, live from the microphone, until someone presses Ctrl+C."""
    log = Logger()
    handles = _launch(cfg, (*WORKERS, "audio"), log)
    code = 0
    try:
        while all(handle.poll() is None for handle in handles):
            time.sleep(0.05)
        codes = [handle.returncode for handle in handles if handle.returncode]
        if codes:
            log.error("stage.exited", codes=codes)
            code = 1
    except KeyboardInterrupt:
        pass
    finally:
        stop(handles, log)
    return code


def stop(handles: list[subprocess.Popen], log: Logger | None = None) -> None:
    for handle in handles:
        handle.terminate()
    for handle in handles:
        try:
            handle.wait(timeout=5)
        except subprocess.TimeoutExpired:
            (log or Logger()).warning("stage.forcequit", pid=handle.pid)
            handle.kill()
            handle.wait()
        except KeyboardInterrupt:
            continue


def _launch(
    cfg: Config, stages: tuple[str, ...], log: Logger | None = None
) -> list[subprocess.Popen]:
    cfg.network.session = uuid4().hex[:8]
    cfg.prepare()
    log = log or Logger()
    env = {
        **os.environ,
        "KAI_CLIENT_CONFIG": str(dump(cfg, Path(cfg.network.runtime_dir) / "launch.toml")),
    }
    handles = [
        subprocess.Popen(
            [sys.executable, "-m", "kai_client.main", "--stage", stage], env=env
        )
        for stage in stages
    ]
    log.step(
        "session.start",
        session=cfg.network.session,
        pids={stage: handle.pid for stage, handle in zip(stages, handles, strict=True)},
        config=env["KAI_CLIENT_CONFIG"],
    )
    return handles
