from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from time import monotonic_ns
from typing import Any

from kai_client.llm import LlmStream
from kai_client.llm.chunking import SentenceBatcher, speakable
from kai_client.runtime.bus import Stream
from kai_client.runtime.protocol import (
    FLAG_AUDIO,
    FLAG_FINAL,
    FLAG_FIRST,
    FLAG_PARTIAL,
    Kind,
    Msg,
    decode,
    encode,
    encode_info,
)
from kai_client.runtime.stage import Stage
from kai_client.utils.config import Config

END_OF_RESPONSE = b""


@dataclass
class Response:
    source: str
    turn: int
    epoch: int
    live: bool = True
    prefill: bool = False
    audio: tuple[bytes, int] | None = None
    buffered: list[str] = field(default_factory=list)
    replay: str = ""
    spoken: list[str] = field(default_factory=list)
    emitted: int = 0
    failed: str = ""
    marks: list[tuple[str, int]] = field(default_factory=list)
    task: asyncio.Task[None] | None = None
    finished: asyncio.Event = field(default_factory=asyncio.Event)


class Composer(Stage):
    name = "llm"

    def __init__(self, cfg: Config, llm: LlmStream) -> None:
        super().__init__(cfg)
        self.turns = Stream(cfg.network.stream("turn"), bind=False)
        self.sentences_out = Stream(cfg.network.stream("sentence"), bind=True)
        self.llm = llm
        self.backends = (llm,)
        self.chat: list[dict[str, Any]] = []
        self.pending: Response | None = None
        self.active: Response | None = None
        self._said: tuple[int, int, str] = (0, 0, "")
        self._since = 0

    async def start(self) -> None:
        await super().start()
        self.tasks.append(asyncio.create_task(self._turn_inbox(), name="llm.turns"))
        self.tasks.append(asyncio.create_task(self.llm.prewarm(), name="llm.prewarm"))

    async def stop(self) -> None:
        for run in (self.pending, self.active):
            if run and run.task and not run.task.done():
                run.task.cancel()
        await super().stop()

    def state(self) -> dict[str, Any]:
        return {
            "generating": self.active is not None and not self.active.finished.is_set()
        }

    def on_cancel(self, msg: Msg) -> None:
        if not self.guard.adopt(msg.turn, msg.epoch):
            return
        self._discard_pending()
        if self.active is not None and not self.active.finished.is_set():
            self.active.task and self.active.task.cancel()

    async def _turn_inbox(self) -> None:
        while not self.stopping.is_set():
            try:
                frames = await self.turns.recv()
            except asyncio.CancelledError:
                return
            msg = decode(frames)
            if msg.kind is not Kind.TURN or not self.guard.accept(msg):
                self.log.detail("turn.stale", msg.turn, msg.epoch, kind=msg.kind.name)
                continue
            error = msg.json.get("error")
            if error:
                self._report(msg.turn, msg.epoch, error)
                continue
            key, audio = self._input(msg)
            if not key:
                self.log.warning(
                    "turn.empty",
                    turn=msg.turn,
                    epoch=msg.epoch,
                    partial=bool(msg.flags & FLAG_PARTIAL),
                    audio=bool(msg.flags & FLAG_AUDIO),
                )
                continue
            partial = bool(msg.flags & FLAG_PARTIAL)
            if partial:
                self.log.detail(
                    "turn.receive", msg.turn, msg.epoch, source=str(key)[:120]
                )
            else:
                self.log.step(
                    "turn.receive",
                    msg.turn,
                    msg.epoch,
                    source=str(key)[:120],
                    bytes=len(audio[0]) if audio else 0,
                )
            if partial:
                if self.cfg.llm.speculate:
                    self._warm(msg, key, audio)
            else:
                await self._commit(msg, key, audio)

    @staticmethod
    def _input(msg: Msg) -> tuple[str, tuple[bytes, int] | None]:
        payload = msg.json
        if msg.flags & FLAG_AUDIO:
            return payload["audio_id"], (msg.audio, payload["rate"])
        return payload["text"].strip(), None

    def _warm(self, msg: Msg, key: str, audio: tuple[bytes, int] | None) -> None:
        seen = (msg.turn, msg.epoch, key)
        now = monotonic_ns()
        if seen != self._said:
            opening = self._said[:2] != (msg.turn, msg.epoch)
            self._said, self._since = seen, now
            self._speculate(
                msg,
                key,
                audio,
                prefill=self.cfg.llm.prefill_warm and not opening and audio is None,
            )
            return
        if self.pending is not None and not self.pending.prefill:
            self.log.detail("warm.same", msg.turn, msg.epoch)
            return
        if (
            now - self._since >= 2 * int(self.cfg.vad.partial_every_ms * 1e6)
            and self.pending is not None
        ):
            self._speculate(msg, key, audio, prefill=False)

    def _speculate(
        self, msg: Msg, key: str, audio: tuple[bytes, int] | None, prefill: bool
    ) -> None:
        self._discard_pending()
        line = self.log.detail if prefill else self.log.step
        line("warm.start", msg.turn, msg.epoch, source=str(key)[:120])
        run = Response(
            source=key,
            turn=msg.turn,
            epoch=msg.epoch,
            live=False,
            prefill=prefill,
            audio=audio,
        )
        run.task = asyncio.create_task(
            self._generate(run),
            name=f"{'prefill' if prefill else 'prefetch'}:{msg.turn}",
        )
        self.pending = run

    async def _commit(
        self, msg: Msg, key: str, audio: tuple[bytes, int] | None
    ) -> None:
        run = self.pending
        self.pending = None
        if run is not None and run.prefill:
            self.log.detail(
                "prefill.held", run.turn, run.epoch, warmed=str(run.source)[:120]
            )
            run = None
        if run is not None and run.source == key:
            self.log.step(
                "warm.kept",
                run.turn,
                run.epoch,
                emitted=run.emitted,
                failed=run.failed or None,
            )
            run.live = True
            if run.failed and not run.emitted:
                self._report(run.turn, run.epoch, run.failed)
            self._publish_marks(run)
            self._release(run)
            if run.task is None or run.task.done():
                await self._close(run)
            else:
                self.active = run
            return
        if run is not None:
            self.log.step(
                "warm.thrown",
                msg.turn,
                msg.epoch,
                warmed=str(run.source)[:120],
                said=str(key)[:120],
            )
            if run.task is not None:
                run.task.cancel()
        fresh = Response(source=key, turn=msg.turn, epoch=msg.epoch, audio=audio)
        fresh.task = asyncio.create_task(
            self._generate(fresh), name=f"generate:{msg.turn}"
        )
        self.active = fresh

    def _discard_pending(self) -> None:
        run, self.pending = self.pending, None
        if run is not None and run.task is not None:
            self.log.detail("warm.cancelled", run.turn, run.epoch)
            run.task.cancel()

    def _mark(self, run: Response, point: str) -> None:
        stamp = monotonic_ns()
        if run.live:
            self.tracer.mark(point, run.turn, run.epoch, ns=stamp)
        else:
            run.marks.append((point, stamp))

    def _publish_marks(self, run: Response) -> None:
        for point, stamp in run.marks:
            self.tracer.mark(point, run.turn, run.epoch, ns=stamp)
        run.marks.clear()

    def _release(self, run: Response) -> None:
        if run.replay:
            self.events_out.send(
                encode(Kind.TOKEN, run.turn, run.epoch, 0, run.replay.encode())
            )
            run.replay = ""
        for sentence in run.buffered:
            self._emit(run, sentence)
        run.buffered.clear()

    async def _generate(self, run: Response) -> None:
        config = self.cfg.llm
        self._drop_unheard()
        messages = [
            {"role": "system", "content": config.system_prompt},
            *self.chat,
            {"role": "user", "content": _content(run)},
        ]
        if run.prefill:
            try:
                async for _ in self.llm.stream(messages, max_tokens=1):
                    pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.log.detail(
                    "prefill.failed",
                    run.turn,
                    run.epoch,
                    error=f"{type(exc).__name__}: {exc}",
                )
            return
        batcher = SentenceBatcher(
            first_batch=config.first_batch_sentences,
            batch=config.batch_sentences,
            min_first_chars=config.min_first_chars,
            max_chunk_chars=config.max_chunk_chars,
        )
        self._mark(run, "llm_start")
        first = True
        try:
            async for delta in self.llm.stream(messages):
                if self._prefetched(run):
                    break
                if first:
                    first = False
                    self._mark(run, "llm_first")
                if run.live:
                    self.events_out.send(
                        encode(Kind.TOKEN, run.turn, run.epoch, 0, delta.encode())
                    )
                else:
                    run.replay += delta
                for sentence in batcher.push(delta):
                    self._sentence(run, sentence)
            for sentence in batcher.finish():
                self._sentence(run, sentence)
        except asyncio.CancelledError:
            self._mark(run, "llm_end")
            if run.live:
                await self._close(run, cancelled=True)
            raise
        except Exception as exc:  # noqa: BLE001
            run.failed = f"{type(exc).__name__}: {exc}"
            self.log.failure("llm.failed", exc, run.turn, run.epoch, warm=not run.live)
            if run.live:
                self._report(run.turn, run.epoch, run.failed)
        self._mark(run, "llm_end")
        if run.live:
            await self._close(run)

    def _prefetched(self, run: Response) -> bool:
        return not run.live and len(run.buffered) >= self.cfg.llm.prefetch_max_sentences

    def _sentence(self, run: Response, sentence: str) -> None:
        cleaned = " ".join(speakable(sentence).split())
        if not cleaned:
            return
        if run.live:
            self._emit(run, cleaned)
        elif not self._prefetched(run):
            run.buffered.append(cleaned)

    def _report(self, turn: int, epoch: int, error: str) -> None:
        """A stage that answered nothing must still say why: the audio boundary prints it and the turn ends."""
        self.log.error("turn.failed", turn=turn, epoch=epoch, error=error)
        self.events_out.send(encode_info(Kind.TOKEN, turn, epoch, {"error": error}))

    def _emit(self, run: Response, text: str) -> None:
        run.emitted += 1
        run.spoken.append(text)
        flags = FLAG_FIRST if run.emitted == 1 else 0
        self.log.step(
            "sentence.emit",
            run.turn,
            run.epoch,
            n=run.emitted,
            warm=not run.live,
            text=text,
        )
        self.sentences_out.send(
            encode(
                Kind.SENTENCE, run.turn, run.epoch, run.emitted, text.encode(), flags
            )
        )

    async def _close(self, run: Response, cancelled: bool = False) -> None:
        if run.emitted:
            end = encode(
                Kind.SENTENCE,
                run.turn,
                run.epoch,
                run.emitted + 1,
                END_OF_RESPONSE,
                FLAG_FINAL,
            )
            self.sentences_out.send(end)
        reply = " ".join(run.spoken).strip()
        self.log.step(
            "turn.close",
            run.turn,
            run.epoch,
            sentences=run.emitted,
            cut=cancelled or None,
        )
        self.chat.append({"role": "user", "content": _content(run)})
        if reply:
            self.chat.append({"role": "assistant", "content": reply})
        keep = self.cfg.llm.history_turns * 2
        del self.chat[:-keep]
        run.finished.set()
        if self.active is run:
            self.active = None

    def _drop_unheard(self) -> None:
        """A turn whose audio has left the window was never said: it leaves with its own reply.

        Leaving a marker in its place puts a claim about a missing file in front of the model, and
        it answers that instead of listening to the turn that really is in front of it.
        """
        keep = max(0, self.cfg.llm.audio_history_turns - 1)
        audio = [
            index
            for index, entry in enumerate(self.chat)
            if isinstance(entry["content"], list)
        ]
        unheard = set(audio[: len(audio) - keep])
        self.chat[:] = [
            entry
            for index, entry in enumerate(self.chat)
            if index not in unheard
            and not (entry["role"] == "assistant" and index - 1 in unheard)
        ]


def _content(run: Response) -> str | list[dict]:
    if run.audio is None:
        return run.source
    pcm, rate = run.audio
    return [{"type": "audio", "pcm": pcm, "rate": rate}]
