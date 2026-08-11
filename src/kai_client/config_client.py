from kai_shared.config_shared import SharedConfig
from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class DummySettings(BaseModel):
    test: str = "hello world"


class ClientConfig(BaseSettings):
    model_config = SettingsConfigDict(toml_file="config.toml")
    shared: SharedConfig = SharedConfig()
    dummy_config: DummySettings = DummySettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls),)


settings_client = ClientConfig()
