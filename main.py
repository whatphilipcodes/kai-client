import asyncio
import json
import queue
import threading
import time

import numpy as np
import sounddevice as sd
import uvloop
from kai_shared.io.node import PipelineNode
from kai_shared.schemata.ipc import AudioStreamMetadata, TokenStreamMetadata
from kai_shared.utils.logger import get_logger, setup_logging
from pydantic import ValidationError

from src.kai_client.config_client import settings_client

logger = get_logger(__name__)


class ClientNode(PipelineNode):
    def __init__(self, config):
        super().__init__(config)
        self.audio_queue = queue.Queue()
        self.prebuffer_chunks = 3
        self.stream_config = None
        self.playback_thread = threading.Thread(
            target=self._playback_worker, daemon=True
        )
        self.input_thread = threading.Thread(target=self._input_worker, daemon=True)
        self.loop = None

    async def start(self):
        self.loop = asyncio.get_running_loop()
        await super().start()
        self.playback_thread.start()
        self.input_thread.start()

    def _playback_worker(self):
        while self._running:
            if self.stream_config is None:
                time.sleep(0.05)
                continue

            sr, channels, dtype = self.stream_config
            try:
                with sd.OutputStream(
                    samplerate=sr, channels=channels, dtype=dtype
                ) as stream:
                    while self.audio_queue.qsize() < self.prebuffer_chunks:
                        time.sleep(0.05)

                    while self._running and self.stream_config is not None:
                        try:
                            chunk = self.audio_queue.get(timeout=1.0)
                            if chunk is None:
                                self.stream_config = None
                                break
                            stream.write(chunk)
                        except queue.Empty:
                            continue
            except Exception as e:
                logger.error(f"Audio stream error: {e}")
                self.stream_config = None

    def _input_worker(self):
        while self._running:
            try:
                user_input = input("\nInput: ")
            except KeyboardInterrupt, EOFError:
                break

            if user_input.strip().lower() == "exit":
                self.stop()
                break

            if not user_input.strip():
                continue

            req_id = "req-" + str(time.time())
            meta = TokenStreamMetadata(request_id=req_id, is_final=True)
            meta_json = meta.model_dump_json().encode("utf-8")
            meta_len = len(meta_json).to_bytes(4, byteorder="big")
            raw_data = user_input.encode("utf-8")
            payload = meta_len + meta_json + raw_data

            logger.info(f"Dispatching prompt to LLM: '{user_input}'")

            if self.loop:
                asyncio.run_coroutine_threadsafe(self.send_reliable(payload), self.loop)

    async def handle_reliable(self, payload: bytes) -> None:
        meta_len = int.from_bytes(payload[:4], byteorder="big")
        meta_json_str = payload[4 : 4 + meta_len].decode("utf-8")

        try:
            meta_dict = json.loads(meta_json_str)
            if meta_dict.get("stream_type") != "audio":
                return
            meta = AudioStreamMetadata(**meta_dict)
        except ValidationError, json.JSONDecodeError:
            return

        if self.stream_config is None and not meta.is_final:
            self.stream_config = (meta.sample_rate, meta.channels, meta.dtype)

        audio_data = payload[4 + meta_len :]

        if meta.is_final:
            logger.info("Received final audio stream signal.")
            self.audio_queue.put(None)
        elif len(audio_data) > 0:
            logger.info(
                f"Received audio chunk: {len(audio_data)} bytes | dtype: {meta.dtype} | sr: {meta.sample_rate}"
            )
            chunk = np.frombuffer(audio_data, dtype=meta.dtype)
            self.audio_queue.put(chunk)


async def main() -> None:
    setup_logging()
    node = ClientNode(settings_client.shared)
    await node.run()


if __name__ == "__main__":
    uvloop.run(main())
