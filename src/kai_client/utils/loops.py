from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import uvloop


def new_event_loop() -> asyncio.AbstractEventLoop:
    """The loop every place in this project that needs one gets: uvloop's, in C."""
    return uvloop.new_event_loop()


def run(main: Coroutine[Any, Any, Any], *, debug: bool = False) -> Any:
    """`asyncio.run` on uvloop, which is what a stage process and every test of one does."""
    return asyncio.run(main, loop_factory=new_event_loop, debug=debug)
