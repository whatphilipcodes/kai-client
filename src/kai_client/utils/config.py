from __future__ import annotations

import os
import tomllib
from enum import StrEnum
from ipaddress import IPv4Address
from pathlib import Path

import tomli_w
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

IPC_PATH_LIMIT = 103


class LogLevel(StrEnum):
    OFF = "off"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


class Transport(StrEnum):
    IPC = "ipc"
    TCP = "tcp"


class AsrMode(StrEnum):
    REALTIME = "realtime"
    BATCH = "batch"
    DIRECT = "direct"


class VadEngine(StrEnum):
    ENERGY = "energy"
    MANUAL = "manual"


def project_root() -> Path:
    """The checkout holding `config.toml` and `assets/`, found by walking up for `pyproject.toml`.

    Counting parent directories is what broke when the package moved under `src/`: the answer
    depends on how deep this file sits, which is not a fact worth carrying. A file that identifies
    a project is. A wheel installed outside any checkout has no project to find, and falls back to
    wherever it was started from.
    """
    here = Path(__file__).resolve().parent
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


ROOT = project_root()


def config_file() -> Path:
    """$KAI_CLIENT_CONFIG when the shell names one, else the `config.toml` at the project root."""
    named = os.environ.get("KAI_CLIENT_CONFIG")
    return Path(named).expanduser() if named else ROOT / "config.toml"


def asset(value: str) -> Path:
    """A path written in `config.toml`: taken as it stands when absolute, otherwise from the
    project root, so the command finds its files whoever it was invoked by."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


STREAMS = ("turn", "sentence")
UDP_STREAMS = ("mic", "tts")
STAGES = ("audio", "asr", "llm", "tts")


class Section(BaseModel):
    """One table of config.toml. A key that is not a field is a mistake, not an extra option."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Endpoint(Section):
    """One socket, addressed either as a path on this machine or as a port on a host."""

    protocol: Transport = Transport.IPC
    host: str | IPv4Address = "127.0.0.1"
    path: str = ""
    port: int = 0

    @property
    def address(self) -> str:
        return (
            f"ipc://{self.path}"
            if self.protocol is Transport.IPC
            else f"tcp://{self.host}:{self.port}"
        )

    @property
    def socket(self) -> str:
        return (
            self.path if self.protocol is Transport.IPC else f"{self.host}:{self.port}"
        )


class System(Section):
    log_level: LogLevel = LogLevel.WARNING


class Network(Section):
    transport: Transport = Transport.IPC
    ipc_dir: str = "/tmp/kai-client"
    host: str | IPv4Address = "127.0.0.1"
    zmq_base: int = 5560
    udp_base: int = 5570
    connect_grace_ms: int = 300
    session: str = "default"

    def stream(self, name: str) -> Endpoint:
        return self._endpoint(
            f"{self.session}-{name}", self.zmq_base + STREAMS.index(name)
        )

    def events(self, stage: str) -> Endpoint:
        return self._endpoint(
            f"{self.session}-ev-{stage}", self.zmq_base + 10 + STAGES.index(stage)
        )

    def udp(self, name: str) -> Endpoint:
        return self._endpoint("", self.udp_base + UDP_STREAMS.index(name))

    def endpoints(self) -> list[Endpoint]:
        return [self.stream(name) for name in STREAMS] + [
            self.events(stage) for stage in STAGES
        ]

    def _endpoint(self, name: str, port: int) -> Endpoint:
        path = f"{self.ipc_dir}/{name}" if self.transport is Transport.IPC else ""
        return Endpoint(protocol=self.transport, host=self.host, path=path, port=port)


class Audio(Section):
    frame_ms: float = 20.0
    input_rate: int = 16000
    input_device: str = ""
    output_device: str = ""
    jitter_ms: float = 20.0
    prebuffer_ms: float = 0.0
    max_lag_ms: float = 30000.0


class Vad(Section):
    engine: VadEngine = VadEngine.ENERGY
    min_speech_ms: int = 160
    min_silence_ms: int = 240
    rms_open: float = 0.012
    rms_close: float = 0.007
    speech_pad_ms: int = 60
    partial_every_ms: int = 150


class Asr(Section):
    mode: AsrMode = AsrMode.REALTIME
    realtime: str = "ws://127.0.0.1:8090/v1/realtime"
    batch: str = "http://127.0.0.1:8090/v1"
    model: str = "model"
    api_key: str = ""
    language: str = ""

    @property
    def url(self) -> str:
        """The address the mode names; a direct turn never asks for one."""
        return self.batch if self.mode is AsrMode.BATCH else self.realtime


class Llm(Section):
    url: str = "http://127.0.0.1:8080/v1"
    model: str = "model"
    api_key: str = ""
    system_prompt: str = "Du bist Kaspar. Antworte in einem kurzen Satz."
    temperature: float = 0.3
    max_tokens: int = 128
    history_turns: int = 8
    audio_history_turns: int = 1
    prefetch_max_sentences: int = 4
    first_batch_sentences: int = 1
    batch_sentences: int = 3
    min_first_chars: int = 20
    max_chunk_chars: int = 120
    speculate: bool = True
    prefill_warm: bool = True


class Tts(Section):
    url: str = "http://127.0.0.1:8091/v1"
    model: str = "model"
    voice: str = "default"
    api_key: str = ""
    sample_rate: int = 24000
    ref_audio: str = ""
    ref_text: str = ""
    stream_format: str = "audio"
    chunk_ms: float = 60.0


class Trace(Section):
    enabled: bool = True
    quiet: bool = False


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KAI_CLIENT_", env_nested_delimiter="__", extra="forbid"
    )

    system: System = Field(default_factory=System)
    network: Network = Field(default_factory=Network)
    audio: Audio = Field(default_factory=Audio)
    vad: Vad = Field(default_factory=Vad)
    asr: Asr = Field(default_factory=Asr)
    llm: Llm = Field(default_factory=Llm)
    tts: Tts = Field(default_factory=Tts)
    trace: Trace = Field(default_factory=Trace)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # An argument beats an environment variable beats a file: `Config(llm=...)` in a test,
        # `KAI_CLIENT_LLM__URL` in a shell, and the file $KAI_CLIENT_CONFIG names over the
        # one sitting at the project root.
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls, config_file().resolve()),
        )

    @classmethod
    def defaults(cls) -> Config:
        """The shipped values with no file read: what a test starts from, so one person's
        config.toml pointing at a live server cannot quietly become the fixture."""
        return cls(
            system=System(),
            network=Network(),
            audio=Audio(),
            vad=Vad(),
            asr=Asr(),
            llm=Llm(),
            tts=Tts(),
            trace=Trace(),
        )

    def prepare(self) -> None:
        if self.network.transport is not Transport.IPC:
            return
        os.makedirs(self.network.ipc_dir, exist_ok=True)
        longest = max(
            self.network.endpoints(), key=lambda endpoint: len(endpoint.socket)
        )
        if len(longest.socket) > IPC_PATH_LIMIT:
            raise ValueError(
                f"network.ipc_dir is too long for a unix socket: {longest.socket!r} exceeds "
                f'{IPC_PATH_LIMIT} characters; shorten ipc_dir or set transport = "tcp"'
            )


def load(path: str | Path | None = None) -> Config:
    """The file named here, or $KAI_CLIENT_CONFIG, or the config.toml at the project root."""
    if path is None:
        cfg = Config()
    else:
        cfg = Config.model_validate(
            tomllib.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
        )
    cfg.prepare()
    return cfg


def dump(cfg: Config, path: str | Path) -> Path:
    """What a launched stage process reads: the running config, as one toml file."""
    target = Path(path)
    target.write_text(tomli_w.dumps(cfg.model_dump(mode="json")), encoding="utf-8")
    return target
