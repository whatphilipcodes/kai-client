from __future__ import annotations

import argparse

from kai_client.runner import serve
from kai_client.runtime import launch
from kai_client.utils import logger, loops
from kai_client.utils.config import STAGES, load

USAGE = """
The microphone and the speakers are always the input and the output, and every
choice between them - transport, the gate, which server answers each stage, how
sentences are batched - is a key in config.toml. Ctrl+C stops."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kai-client",
        description="Local speech-to-speech: UDP audio, ZeroMQ tokens, one process per stage.",
        epilog=USAGE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stage", choices=STAGES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    cfg = load()
    logger.setup(cfg)
    if args.stage:
        loops.run(serve(cfg, args.stage))
        return 0
    return launch.session(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
