from __future__ import annotations

import asyncio
import sys
from collections.abc import Coroutine
from typing import Any

if sys.platform != "win32":
    import uvloop


def new_event_loop() -> asyncio.AbstractEventLoop:
    if sys.platform != "win32":
        return uvloop.new_event_loop()
    return asyncio.new_event_loop()


def run(main: Coroutine[Any, Any, Any], *, debug: bool = False) -> Any:
    return asyncio.run(main, loop_factory=new_event_loop, debug=debug)
