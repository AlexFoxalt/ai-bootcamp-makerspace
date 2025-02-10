from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )
    llm_model: str
    embeddings_model: str


class CheapSettings(Settings):
    llm_model: str = "gpt-4o-mini"
    embeddings_model: str = "text-embedding-3-small"
    model_dims: int = 1536
    name: str = "cheap"


class ExpensiveSettings(Settings):
    llm_model: str = "gpt-4-turbo"
    embeddings_model: str = "text-embedding-3-large"
    model_dims: int = 3072
    name: str = "expensive"


CHEAP_SETTINGS = CheapSettings()
EXPENSIVE_SETTINGS = ExpensiveSettings()
